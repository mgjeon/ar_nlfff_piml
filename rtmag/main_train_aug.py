import os
import json
import argparse
from pathlib import Path

import numpy as np

import torch
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from neuralop.models import UNO

from rtmag.train.training_aug import train, get_dataloaders


#-----------------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str)
args = parser.parse_args()

with open(args.config) as config:
    info = json.load(config)
    for key, value in info.items():
        args.__dict__[key] = value

base_path = args.base_path
os.makedirs(base_path, exist_ok=True)

np.save(os.path.join(args.base_path, "args.npy"), args.__dict__)

log_dir = Path(args.base_path) / "log"
log_dir.mkdir(parents=True, exist_ok=True)
writer = SummaryWriter(log_dir)

if args.model["model_name"] == "UNO":
    model = UNO(
        in_channels = args.model["in_channels"],
        hidden_channels = args.model["hidden_channels"],
        lifting_channels = args.model["lifting_channels"],
        projection_channels = args.model["projection_channels"],
        out_channels = args.model["out_channels"],

        n_layers = args.model["n_layers"],

        uno_out_channels = args.model["uno_out_channels"],
        uno_scalings = args.model["uno_scalings"],
        uno_n_modes = args.model["uno_n_modes"], 
    )
else:
    raise NotImplementedError

if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)

optimizer = Adam(model.parameters(), lr=args.training['learning_late'])

CHECKPOINT_PATH = os.path.join(args.base_path, "last.pt")

if os.path.exists(CHECKPOINT_PATH):   
    checkpoint = torch.load(CHECKPOINT_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    ck_epoch = checkpoint['epoch'] + 1
else:
    ck_epoch = 0

train_dataloader, test_dataloader = get_dataloaders(args)

train(model, optimizer, train_dataloader, test_dataloader, ck_epoch, CHECKPOINT_PATH, args, writer)