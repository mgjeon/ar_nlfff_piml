import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, RandomSampler

from rtmag.don.dataset_label import DeepONetDatasetCNNlabeldata
from rtmag.don.model import DeepONet

import argparse

from torch.utils.tensorboard import SummaryWriter

from torchmetrics.regression import ConcordanceCorrCoef

#---------------------------------------------------------------------------------------------------------------#

def jacobian(output, coords):
    jac_matrix = [torch.autograd.grad(output[..., i], coords,
                                      grad_outputs=torch.ones_like(output[..., i]).to(output),
                                      retain_graph=True, create_graph=True, allow_unused=True)[0]
                  for i in range(output.shape[-1])]
    jac_matrix = torch.stack(jac_matrix, dim=-1)
    return jac_matrix


def calculate_pde_loss(b, coords):
    jac_matrix = jacobian(b, coords)
    dBx_dx = jac_matrix[..., 0, 0]
    dBy_dx = jac_matrix[..., 1, 0]
    dBz_dx = jac_matrix[..., 2, 0]
    dBx_dy = jac_matrix[..., 0, 1]
    dBy_dy = jac_matrix[..., 1, 1]
    dBz_dy = jac_matrix[..., 2, 1]
    dBx_dz = jac_matrix[..., 0, 2]
    dBy_dz = jac_matrix[..., 1, 2]
    dBz_dz = jac_matrix[..., 2, 2]
    #
    curl_x = dBz_dy - dBy_dz
    curl_y = dBx_dz - dBz_dx
    curl_z = dBy_dx - dBx_dy
    #
    j = torch.stack([curl_x, curl_y, curl_z], -1)
    #
    jxb = torch.cross(j, b, -1)
    loss_ff = torch.sum(jxb ** 2, dim=-1) / (torch.sum(b ** 2, dim=-1) + 1e-7)
    loss_ff = loss_ff.mean()

    loss_div = (dBx_dx + dBy_dy + dBz_dz) ** 2
    loss_div = loss_div.mean()
    return loss_div, loss_ff

#---------------------------------------------------------------------------------------------------------------#

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True)
args = parser.parse_args()

with open(args.config) as config:
    info = json.load(config)
    for key, value in info.items():
        args.__dict__[key] = value

base_path = Path(args.base_path)
os.makedirs(base_path, exist_ok=True)

np.save(os.path.join(args.base_path, "args.npy"), args.__dict__)

log_dir = base_path / "log"
log_dir.mkdir(parents=True, exist_ok=True)
writer = SummaryWriter(log_dir)

#---------------------------------------------------------------------------------------------------------------#
        
dataset_path = args.dataset["dataset_path"]
if dataset_path.endswith('npz'):
    data_path = [dataset_path]
else:
    test_noaa = args.dataset["test_noaa"]
    files = list(Path(dataset_path).glob('**/input/*.npz'))
    if isinstance(test_noaa, str):
        files = sorted([f for f in files if not test_noaa == str(f.parent.parent.stem)])
    elif isinstance(test_noaa, list):
        files = sorted([f for f in files if not test_noaa[0] == str(f.parent.parent.stem)])
        for t_noaa in test_noaa[1:]:
            files = sorted([f for f in files if not t_noaa == str(f.parent.parent.stem)])

    data_path = files

dataset_num = len(data_path)

#---------------------------------------------------------------------------------------------------------------#

trunk_in_dim = args.model["trunk_in_dim"]
out_dim = args.model["out_dim"]
latent_dim = args.model["latent_dim"]
hidden_dim = args.model["hidden_dim"]
num_layers = args.model["num_layers"]

if args.model["name"] == "DeepONet":
    model = DeepONet(trunk_in_dim, 
                     out_dim, 
                     latent_dim, 
                     hidden_dim, 
                     num_layers)
else:
    raise NotImplementedError

#---------------------------------------------------------------------------------------------------------------#

if args.dataset["name"] == "DeepONetDatasetCNNlabeldata":
    cube_shape = args.dataset["cube_shape"]
    b_norm = args.dataset["b_norm"]
    spatial_norm = args.dataset["spatial_norm"]
    bottom_batch_coords = int(args.dataset["bottom_batch_coords"])
    data_batch_coords = int(args.dataset["data_batch_coords"])
    random_batch_coords = int(args.dataset["random_batch_coords"])
    don_dataset = DeepONetDatasetCNNlabeldata(data_path, 
                                              b_norm, spatial_norm, cube_shape,
                                              bottom_batch_coords,
                                              data_batch_coords,
                                              random_batch_coords)
    
else:
    raise NotImplementedError

#---------------------------------------------------------------------------------------------------------------#

batch_size = args.training["batch_size"]
num_workers = args.training["num_workers"]
num_samples = int(args.training["num_samples"])
don_loader = DataLoader(don_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                        sampler=RandomSampler(don_dataset, replacement=True, num_samples=num_samples))

total_iterations = num_samples // batch_size

#---------------------------------------------------------------------------------------------------------------#
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

CHECKPOINT_PATH = base_path / "last.pt"
if os.path.exists(CHECKPOINT_PATH):   
    checkpoint = torch.load(CHECKPOINT_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    ck_idx = 0

optimizer = torch.optim.Adam(model.parameters(), lr=args.training["lr"])

print(f"Start: {ck_idx}")

tqdm_loader = tqdm(don_loader, desc='Train', initial=ck_idx, ncols=140)

best_loss = np.inf
for id, batch in enumerate(tqdm_loader):
    idx = id + ck_idx
    model = model.train()

    branch_input = batch['branch_input']

    bottom_coords = batch['bottom_coords']
    bottom_coords.requires_grad = True
    data_coords = batch['data_coords']
    data_coords.requires_grad = True
    random_coords = batch['random_coords']
    random_coords.requires_grad = True

    n_bottom_coords = bottom_coords.shape[1]
    n_data_coords = data_coords.shape[1]
    coords = torch.concatenate([bottom_coords, data_coords, random_coords], 1)
    
    branch_input = branch_input.to(device)
    coords = coords.to(device)
    b = model(branch_input, coords)

    b_bottom = b[:, :n_bottom_coords, :]
    b_data = b[:, n_bottom_coords:n_bottom_coords+n_data_coords, :]

    b_bottom_true = batch['bottom_values'].to(device)
    b_data_true = batch['data_values'].to(device)

    loss_bc = (b_bottom_true - b_bottom).pow(2).mean()
    loss_mse = (b_data_true - b_data).pow(2).mean()
    ccc = ConcordanceCorrCoef().to(device)
    loss_ccc = torch.abs(1.0 - ccc(b_data_true.flatten(), b_data.flatten()))

    loss_div, loss_ff = calculate_pde_loss(b, coords)

    loss = args.training["lambda_b"] * loss_bc + \
           args.training["lambda_mse"] * loss_mse + \
           args.training["lambda_ccc"] * loss_ccc + \
           args.training["lambda_div"] * loss_div + \
           args.training["lambda_ff"] * loss_ff

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    writer.add_scalar('train/loss', loss.item(), idx)
    writer.add_scalar('train/loss_bc', loss_bc.item(), idx)
    writer.add_scalar('train/loss_mse', loss_mse.item(), idx)
    writer.add_scalar('train/loss_ccc', loss_ccc.item(), idx)
    writer.add_scalar('train/loss_div', loss_div.item(), idx)
    writer.add_scalar('train/loss_ff', loss_ff.item(), idx)

    torch.save({'idx': idx,
                'model_state_dict': model.state_dict(),
                'spatial_norm':spatial_norm,
                'b_norm':b_norm,
                'cube_shape':cube_shape,
                'model_info': {
                    'trunk_in_dim': trunk_in_dim,
                    'out_dim': out_dim,
                    'latent_dim': latent_dim,
                    'hidden_dim': hidden_dim,
                    'num_layers': num_layers
                }}, base_path / "model_last.pt")
    
    torch.save({'idx': idx,
                'model_state_dict': model.state_dict()}, 
                CHECKPOINT_PATH)



    if loss.item() < best_loss:
        torch.save({'idx': idx,
                    'model_state_dict': model.state_dict(),
                    'spatial_norm':spatial_norm,
                    'b_norm':b_norm,
                    'cube_shape':cube_shape,
                    'model_info': {
                        'trunk_in_dim': trunk_in_dim,
                        'out_dim': out_dim,
                        'latent_dim': latent_dim,
                        'hidden_dim': hidden_dim,
                        'num_layers': num_layers
                    }}, base_path / "model_best.pt")
        best_loss = loss.item()
    
    
    if idx % args.training['save_every'] == 0:
        torch.save({'idx': idx,
                    'model_state_dict': model.state_dict(),
                    'spatial_norm':spatial_norm,
                    'b_norm':b_norm,
                    'cube_shape':cube_shape,
                    'model_info': {
                        'trunk_in_dim': trunk_in_dim,
                        'out_dim': out_dim,
                        'latent_dim': latent_dim,
                        'hidden_dim': hidden_dim,
                        'num_layers': num_layers
                    }}, base_path / f"model_{idx}.pt")

    if idx % args.training['val_every'] == 0:
        torch.save({'idx': idx,
                    'model_state_dict': model.state_dict(),
                    'spatial_norm':spatial_norm,
                    'b_norm':b_norm,
                    'cube_shape':cube_shape,
                    'model_info': {
                        'trunk_in_dim': trunk_in_dim,
                        'out_dim': out_dim,
                        'latent_dim': latent_dim,
                        'hidden_dim': hidden_dim,
                        'num_layers': num_layers
                    }}, base_path / f"model_val_{idx}.pt")
        model = model.eval()
        with torch.no_grad():
            ri = np.random.choice(branch_input.shape[0], 1)
            branch_input_single = branch_input[ri]

            bottom_coords = np.stack(np.mgrid[:cube_shape[0], :cube_shape[1], :1], -1).astype(np.float32)
            bottom_coords_shape = bottom_coords.shape
            bottom_coords = bottom_coords.reshape(-1, 3)
            bottom_coords = torch.tensor(bottom_coords, dtype=torch.float32)
            bottom_coords = bottom_coords / spatial_norm

            b_size = 1000
            P = int(np.ceil(bottom_coords.shape[0] / b_size))
            cube = []
            for k in range(P):
                bottom_coord = bottom_coords[k * b_size: (k + 1) * b_size][None, ...]
                bottom_coord = bottom_coord.to(device)
                cube += [model(branch_input_single, bottom_coord).detach().cpu()]
            cube = torch.concatenate(cube, dim=1)[0]
            cube = cube.view(*bottom_coords_shape).numpy()
            b_bottom = cube * b_norm

            branch_input_single = branch_input_single.detach().cpu().numpy().transpose(2, 3, 1, 0)[:, :, 2, 0]*b_norm
            
            fig = plt.figure()
            plt.subplot(2, 1, 1)
            plt.imshow(b_bottom[:, :, 0, 2].T, 
                        origin='lower', cmap='gray')
            plt.colorbar()
            plt.subplot(2, 1, 2)
            plt.imshow(branch_input_single.T, 
                        origin='lower', cmap='gray')
            plt.colorbar()
            writer.add_figure("fig", fig, idx)
            plt.close()