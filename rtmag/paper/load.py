import torch
import numpy as np
import argparse
from pathlib import Path
from neuralop.models import UNO


def load_input_label(meta_path):
    meta_path = Path(meta_path)
    input_files = (meta_path / 'input').glob('*.npz')
    label_files = (meta_path / 'label').glob('*.npz')

    return sorted(list(input_files)), sorted(list(label_files))


class MyModel:
    def __init__(self, meta_path, epoch=None, device=None):
        self.meta_path = meta_path
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.epoch = epoch


    def load_model(self):
        device = self.device
        meta_path = Path(self.meta_path)
        if self.epoch is None:
            checkpoint = torch.load(meta_path / "best_model.pt", map_location=device)
            self.epoch = checkpoint['epoch']
        else:
            try: 
                checkpoint = torch.load(meta_path / f"model_{self.epoch}.pt", map_location=device)
            except:
                checkpoint = torch.load(meta_path / "best_model.pt", map_location=device)
                self.epoch = checkpoint['epoch']

        args = argparse.Namespace()
        info = np.load(meta_path / 'args.npy', allow_pickle=True).item()
        for key, value in info.items():
                args.__dict__[key] = value


        model = UNO(
                hidden_channels = args.model["hidden_channels"],
                in_channels = args.model["in_channels"],
                out_channels = args.model["out_channels"],
                lifting_channels = args.model["lifting_channels"],
                projection_channels = args.model["projection_channels"],
                n_layers = args.model["n_layers"],

                factorization = args.model["factorization"],
                implementation = args.model["implementation"],
                rank = args.model["rank"],

                uno_n_modes = args.model["uno_n_modes"], 
                uno_out_channels = args.model["uno_out_channels"],
                uno_scalings = args.model["uno_scalings"],
            ).to(device)

        checkpoint = torch.load(meta_path / 'best_model.pt')

        model.load_state_dict(checkpoint['model_state_dict'])

        return model, args, checkpoint


    def get_pred(self, input_path):
        model, args, checkpoint = self.load_model()
        device = self.device
        b_norm = args.data["b_norm"]
        model_input = np.load(input_path)['input'].astype(np.float32)
        model_input = torch.from_numpy(model_input)
        model_input = model_input[None, ...]
        # [batch_size, 3, 513, 257, 1]
        model_input = model_input[:, :, :-1, :-1, :]  # remove duplicated periodic boundary
        model_input = model_input.to(device)
        model_input = torch.permute(model_input, (0, 4, 3, 2, 1))
        # [batch_size, 1, 256, 512, 3]
        model_input = model_input / b_norm

        # [batch_size, 256, 256, 512, 3]
        model_output = model(model_input)
        # [512, 256, 256, 3]
        b = model_output.detach().cpu().numpy().transpose(0, 3, 2, 1, 4)[0]
        divi = (b_norm / np.arange(1, b.shape[2] + 1)).reshape(1, 1, -1, 1)
        b = b * divi

        print(f"Model loaded from epoch {self.epoch}")

        return b.astype(np.float32)
    
    def get_pred_from_numpy(self, model_input):
        model, args, checkpoint = self.load_model()
        device = self.device
        b_norm = args.data["b_norm"]
        model_input = torch.from_numpy(model_input)
        model_input = model_input.to(device)
        model_output = model(model_input) / b_norm
        # [512, 256, 256, 3]
        b = model_output.detach().cpu().numpy().transpose(0, 3, 2, 1, 4)[0]
        divi = (b_norm / np.arange(1, b.shape[2] + 1)).reshape(1, 1, -1, 1)
        b = b * divi

        print(f"Model loaded from epoch {self.epoch}")

        return b.astype(np.float32)
    

    def get_label(self, label_path):
        B = np.load(label_path)["label"][:, :-1, :-1, :-1].astype(np.float32)
        B = B.transpose(1, 2, 3, 0)

        return B.astype(np.float32)
    

    def get_pot(self, label_path):
        Bp = np.load(label_path)["pot"][:, :-1, :-1, :-1].astype(np.float32)
        Bp = Bp.transpose(1, 2, 3, 0)

        return Bp.astype(np.float32)
    
    def get_dV(self, input_path):
        inputs = np.load(input_path)
        dx, dy, dz = inputs['dx'], inputs['dy'], inputs['dz']  # Mm
        dx, dy, dz = dx * 1e8, dy * 1e8, dz * 1e8  # cm
        dV = dx * dy * dz # cm^3
        dV = dV.astype(np.float32)
        
        return dx, dy, dz, dV
