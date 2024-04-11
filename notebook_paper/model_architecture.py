import torch
import numpy as np
import argparse
from pathlib import Path
from neuralop.models import UNO

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

meta_path = Path("/home/mgjeon/space/workspace/base/uno_pi_cc_hnorm_unit_aug/best_model.pt")
checkpoint = torch.load(meta_path, map_location=device)

args = argparse.Namespace()
info = np.load(meta_path.parent / 'args.npy', allow_pickle=True).item()
for key, value in info.items():
    args.__dict__[key] = value

model = UNO(
            hidden_channels = 32,
            in_channels = 1,
            out_channels = 256,
            lifting_channels = 256,
            projection_channels = 256,
            n_layers = 6,

            # factorization = "tucker",
            # implementation = args.model["implementation"],
            # rank = args.model["rank"],

            uno_n_modes = [[16,16, 16],
                            [ 8, 8,  8],
                            [ 8, 8,  8],
                            [ 8, 8,  8],
                            [ 8, 8,  8],
                            [16,16, 16]],
            uno_out_channels = [32,
                                64,
                                64,
                                64,
                                64,
                                32],
            uno_scalings = [[1.0,1.0,1.0],
                            [0.5,0.5,0.5],
                            [1.0,1.0,1.0],
                            [1.0,1.0,1.0],
                            [2.0,2.0,2.0],
                            [1.0,1.0,1.0]],
        ).to(device)

x = torch.rand(1, 1, 256, 512, 3).to(device)

y = model(x)