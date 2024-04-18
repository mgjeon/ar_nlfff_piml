import os
import gc

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch 
from torch.utils.data import DataLoader
from torchmetrics.regression import ConcordanceCorrCoef

from rtmag.dataset.dataset_unit import ISEEDataset_Multiple_Unit, ISEEDataset_Unit

from rtmag.train.diff_torch_batch import curl, divergence
from rtmag.test.eval_plot import plot_sample
from rtmag.test import eval

if torch.cuda.is_available():
    device = torch.device("cuda")

#---------------------------------------------------------------------------------------
def criterion(outputs, labels, dx, dy, dz, args):

    loss = {}
    # [b, z, y, x, 3]
    opts = torch.flatten(outputs)
    labs = torch.flatten(labels)

    # mse loss
    loss['mse'] = torch.mean(torch.square(opts - labs))

    # ccc loss
    ccc = ConcordanceCorrCoef().to(device)
    loss['ccc'] = torch.abs(1.0 - ccc(opts, labs))

    # [b, z, y, x, 3] -> [b, x, y, z, 3]
    b = torch.permute(outputs, (0, 3, 2, 1, 4))
    B = torch.permute(labels, (0, 3, 2, 1, 4))

    # boundary condition loss
    # bottom (z=0)
    loss['bc'] = torch.mean(torch.square(b[:, :, :, 0, :] - B[:, :, :, 0, :]))

    # force-free loss
    bx, by, bz = b[..., 0], b[..., 1], b[..., 2]
    jx, jy, jz = curl(bx, by, bz, dx, dy, dz)
    j = torch.stack([jx, jy, jz], -1)

    jxb = torch.cross(j, b, -1)
    loss_ff = (jxb**2).sum(-1) / ((b**2).sum(-1) + 1e-7)
    loss_ff = torch.mean(loss_ff)
    loss['ff'] = loss_ff

    # divergence-less loss
    div_b = divergence(bx, by, bz, dx, dy, dz)
    loss_div = torch.mean(torch.square(div_b))
    loss['div'] = loss_div

    return loss

#---------------------------------------------------------------------------------------
def get_dataloaders(args):
    if args.data["dataset_name"] == "Unit":
        train_dataset = ISEEDataset_Multiple_Unit(args.data['dataset_path'], args.data["b_norm"], test_noaa=args.data['test_noaa'])
        test_dataset = ISEEDataset_Unit(args.data['test_path'], args.data["b_norm"])
    else:
        raise NotImplementedError
    
    train_dataloder = DataLoader(train_dataset, batch_size=args.data['batch_size'], shuffle=True, 
                                 num_workers=args.data["num_workers"], pin_memory=True, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, 
                                 num_workers=args.data["num_workers"], pin_memory=True)

    return train_dataloder, test_dataloader


#---------------------------------------------------------------------------------------
def shared_step(model, sample_batched, args):
    # input --------------------------------------------------------------
    # [b, 3, x, y, 1]
    inputs = sample_batched['input'].to(device)
    # [b, 3, x, y, 1] -> [b, 1, y, x, 3]
    inputs = torch.permute(inputs, (0, 4, 3, 2, 1))

    # predict ------------------------------------------------------------
    # [b, 1, y, x, 3] -> [b, 256, y, x, 3]
    outputs = model(inputs).to(device)

    # label --------------------------------------------------------------
    # [b, 3, x, y, z]
    labels = sample_batched['label'].to(device)

    # [b, 3, x, y, z] -> [b, z, y, x, 3]
    labels = torch.permute(labels, (0, 4, 3, 2, 1))

    # [b]
    dx = sample_batched['dx'].flatten().to(device)
    dy = sample_batched['dy'].flatten().to(device)
    dz = sample_batched['dz'].flatten().to(device)

    # [b, 3, x, y, z]

    loss = criterion(outputs, labels, dx, dy, dz, args)

    return loss


#---------------------------------------------------------------------------------------
def train(model, optimizer, train_dataloader, test_dataloader, ck_epoch, CHECKPOINT_PATH, args, writer):
    print("training3")
    model = model.to(device)
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.cuda()

    base_path = args.base_path
    n_epochs = args.training['n_epochs']
    
    global_step_tmp = len(train_dataloader) * ck_epoch
    validation_loss = np.inf
    for epoch in range(ck_epoch, n_epochs+1):

        if epoch == 0:
            with torch.no_grad():
                model = model.eval()
                val_plot(model, test_dataloader, -1, args, writer)

        # Training
        model = model.train()
        total_train_loss = 0
        total_train_loss_mse = 0
        total_train_loss_ccc = 0
        total_train_loss_bc = 0
        total_train_loss_ff = 0
        total_train_loss_div = 0

        with tqdm(train_dataloader, desc='Training', ncols=140) as tqdm_loader_train:
            for i_batch, sample_batched in enumerate(tqdm_loader_train):
                gc.collect()
                torch.cuda.empty_cache()

                tqdm_loader_train.set_description(f"epoch {epoch}")

                loss_dict = shared_step(model, sample_batched, args)
                loss = args.training['w_mse']*loss_dict['mse'] \
                     + args.training['w_ccc']*loss_dict['ccc'] \
                     + args.training['w_bc']*loss_dict['bc'] \
                     + args.training['w_ff']*loss_dict['ff'] \
                     + args.training['w_div']*loss_dict['div']

                loss.backward()

                if ((i_batch + 1) % args.training["num_accmulation_steps"] == 0) or (i_batch + 1 == len(train_dataloader)):
                    optimizer.step()
                    optimizer.zero_grad()

                global_step = global_step_tmp + epoch + i_batch
                tqdm_loader_train.set_postfix(step=global_step)

                total_train_loss += loss.item()
                total_train_loss_mse += loss_dict['mse'].item()
                total_train_loss_ccc += loss_dict['ccc'].item()
                total_train_loss_bc += loss_dict['bc'].item()
                total_train_loss_ff += loss_dict['ff'].item()
                total_train_loss_div += loss_dict['div'].item()

                writer.add_scalar('step_train/loss', loss.item(), global_step)
                writer.add_scalar('step_train/loss_mse', loss_dict['mse'].item(), global_step)
                writer.add_scalar('step_train/loss_ccc', loss_dict['ccc'].item(), global_step)
                writer.add_scalar('step_train/loss_bc', loss_dict['bc'].item(), global_step)
                writer.add_scalar('step_train/loss_ff', loss_dict['ff'].item(), global_step)
                writer.add_scalar('step_train/loss_div', loss_dict['div'].item(), global_step)

                writer.add_scalar('epoch', epoch, global_step)
                
                torch.save({'epoch': epoch, 'global_step': global_step, 
                            'model_state_dict': model.state_dict()}, 
                            os.path.join(args.base_path, "last_model.pt"))

        total_train_loss /= len(train_dataloader)
        total_train_loss_mse /= len(train_dataloader)
        total_train_loss_ccc /= len(train_dataloader)
        total_train_loss_bc /= len(train_dataloader)
        total_train_loss_ff /= len(train_dataloader)
        total_train_loss_div /= len(train_dataloader)

        writer.add_scalar('train/loss', total_train_loss, epoch)
        writer.add_scalar('train/loss_mse', total_train_loss_mse, epoch)
        writer.add_scalar('train/loss_ccc', total_train_loss_ccc, epoch)
        writer.add_scalar('train/loss_bc', total_train_loss_bc, epoch)
        writer.add_scalar('train/loss_ff', total_train_loss_ff, epoch)
        writer.add_scalar('train/loss_div', total_train_loss_div, epoch)
        
        global_step_tmp = global_step

        torch.save({'epoch': epoch, 'global_step': global_step, 
                    'model_state_dict': model.state_dict(), 
                    'optimizer_state_dict': optimizer.state_dict()}, 
                    CHECKPOINT_PATH)
        
        if epoch % args.training['save_epoch_every'] == 0:
            torch.save({'epoch': epoch, 'global_step': global_step,
                        'model_state_dict': model.state_dict()}, os.path.join(base_path, f"model_{epoch}.pt"))

        # Validation
        with torch.no_grad():
            model = model.eval()
            val_plot(model, test_dataloader, epoch, args, writer)
            total_val_loss = val(model, test_dataloader, epoch, args, writer)

        if os.path.exists(os.path.join(base_path, "best_model.pt")):
            checkpoint = torch.load(os.path.join(base_path, "best_model.pt"))
            validation_loss = checkpoint['validation_loss']

        if total_val_loss < validation_loss:
            validation_loss = total_val_loss
            torch.save({'epoch': epoch, 'global_step': global_step, 'validation_loss': validation_loss,
                        'model_state_dict': model.state_dict()}, os.path.join(base_path, "best_model.pt"))


#---------------------------------------------------------------------------------------
def eval_plots(b_pred, b_true, b_pot, func, name):
    heights = np.arange(b_pred.shape[2])

    plots_b = []
    for i in range(b_pred.shape[-2]):
        plots_b.append(func(b_pred[:, :, i, :], b_true[:, :, i, :]))

    plots_B_pot = []
    for i in range(b_pot.shape[-2]):
        plots_B_pot.append(func(b_pot[:, :, i, :], b_true[:, :, i, :]))

    fig = plt.figure(figsize=(6, 8))
    plt.plot(plots_b, heights, color='red', label='PINO')
    plt.plot(plots_B_pot, heights, color='black', label='Potential')
    plt.legend()
    plt.xlabel(name)
    plt.ylabel('height [pixel]')
    plt.xscale('log')
    plt.yscale('linear')
    plt.grid()
    plt.tight_layout()
    return fig

#---------------------------------------------------------------------------------------
def val_plot(model, test_dataloader, epoch, args, writer):
    with torch.no_grad():
        batch = next(iter(test_dataloader))

        if args.data["b_norm"]:
            b_norm = args.data["b_norm"]
        else:
            b_norm = batch['b_norm'].detach().cpu().numpy()
            b_norm = b_norm[0]

        # [b, 3, x, y, 1]
        inputs = batch['input'].to(device)
        # [b, 3, x, y, 1] -> [b, 1, y, x, 3]
        inputs = torch.permute(inputs, (0, 4, 3, 2, 1))
        # [b, 1, y, x, 3] -> [b, z, y, x, 3]
        outputs = model(inputs)

        # [b, z, y, x, 3] -> [x, y, z, 3]
        b_pred = outputs.detach().cpu().numpy()
        b_pred = b_pred[0, ...].transpose(2, 1, 0, 3)

        # [b, 3, x, y, z] -> [x, y, z, 3]
        b_true = batch['label'].detach().cpu().numpy()
        b_true = b_true[0, ...].transpose(1, 2, 3, 0)

        # unnormalize
        b_pred = b_pred * b_norm
        b_true = b_true * b_norm

        fig1, fig2 = plot_sample(b_pred, b_true, ret=True, v_mm=b_norm/2)
        writer.add_figure('plot/pred', fig1, epoch)
        writer.add_figure('plot/true', fig2, epoch)
        plt.close()
        
        #-----------------------------------------------------------
        b_pot = batch['pot'].detach().cpu().numpy()
        b_pot = b_pot[0, ...].transpose(1, 2, 3, 0)
    
        fig = eval_plots(b_pred, b_true, b_pot, eval.l2_error, 'rel_l2_err')
        writer.add_figure(f'plot/rel_l2_err', fig, epoch)
        plt.close()

        fig = eval_plots(b_pred, b_true, b_pot, eval.eps, 'eps')
        writer.add_figure(f'plot/eps', fig, epoch)
        plt.close()

        gc.collect()
        torch.cuda.empty_cache()


#---------------------------------------------------------------------------------------
def val(model, test_dataloader, epoch, args, writer):
    with torch.no_grad():

        total_val_loss = 0.0
        total_val_loss_mse = 0.0
        total_val_loss_ccc = 0.0
        total_val_loss_bc = 0.0
        total_val_loss_ff = 0.0
        total_val_loss_div = 0.0

        for i_batch, sample_batched in enumerate(tqdm(test_dataloader, position=1, desc='Validation', leave=False, ncols=70)):
            gc.collect()
            torch.cuda.empty_cache()
            
            val_loss_dict = shared_step(model, sample_batched, args)

            val_loss = args.training['w_mse']*val_loss_dict['mse'] \
                     + args.training['w_ccc']*val_loss_dict['ccc'] \
                     + args.training['w_bc']*val_loss_dict['bc'] \
                     + args.training['w_ff']*val_loss_dict['ff'] \
                     + args.training['w_div']*val_loss_dict['div']

            total_val_loss += val_loss.item()
            total_val_loss_mse += val_loss_dict['mse'].item()
            total_val_loss_ccc += val_loss_dict['ccc'].item()
            total_val_loss_bc += val_loss_dict['bc'].item()
            total_val_loss_ff += val_loss_dict['ff'].item()
            total_val_loss_div += val_loss_dict['div'].item()
        
        total_val_loss /= len(test_dataloader)
        total_val_loss_mse /= len(test_dataloader)
        total_val_loss_ccc /= len(test_dataloader)
        total_val_loss_bc /= len(test_dataloader)
        total_val_loss_ff /= len(test_dataloader)
        total_val_loss_div /= len(test_dataloader)

        writer.add_scalar('val/loss', total_val_loss, epoch)
        writer.add_scalar('val/loss_mse', total_val_loss_mse, epoch)
        writer.add_scalar('val/loss_ccc', total_val_loss_ccc, epoch)
        writer.add_scalar('val/loss_bc', total_val_loss_bc, epoch)
        writer.add_scalar('val/loss_ff', total_val_loss_ff, epoch)
        writer.add_scalar('val/loss_div', total_val_loss_div, epoch)

        return total_val_loss