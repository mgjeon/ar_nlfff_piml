import os
import json
import argparse
from pathlib import Path

import numpy as np

import torch
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from neuralop.models import UNO

from rtmag.train.training import train, get_dataloaders


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
    if args.model.get("factorization") is not None:
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
        )
    else:
        print("No factorization")
        model = UNO(
            hidden_channels = args.model["hidden_channels"],
            in_channels = args.model["in_channels"],
            out_channels = args.model["out_channels"],
            lifting_channels = args.model["lifting_channels"],
            projection_channels = args.model["projection_channels"],
            n_layers = args.model["n_layers"],

            uno_n_modes = args.model["uno_n_modes"], 
            uno_out_channels = args.model["uno_out_channels"],
            uno_scalings = args.model["uno_scalings"],
        )
else:
    raise NotImplementedError

if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)

optimizer = Adam(model.parameters(), lr=args.training['learning_late'])


try:
    CHECKPOINT_PATH = args.meta_path
    checkpoint = torch.load(CHECKPOINT_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    ck_epoch = checkpoint['epoch'] + 1

    print(f"Loaded checkpoint from {CHECKPOINT_PATH}")
    print(f"Starting from epoch {ck_epoch}")

except:
    CHECKPOINT_PATH = os.path.join(args.base_path, "last.pt")

    if os.path.exists(CHECKPOINT_PATH):   
        checkpoint = torch.load(CHECKPOINT_PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        ck_epoch = checkpoint['epoch'] + 1
    else:
        ck_epoch = 0

train_dataloader, val_dataloader = get_dataloaders(args)

if args.training.get("end_learning_rate") is not None:
    print("scheduler")
    lr_start = args.training['learning_late']
    lr_end = args.training['end_learning_rate']
    lr_decay = args.training['decay_epoch']
    lr_gamma = (lr_end / lr_start) ** (1 / lr_decay) 
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_gamma)
    train(model, optimizer, train_dataloader, val_dataloader, ck_epoch, CHECKPOINT_PATH, args, writer, scheduler)
else:
    train(model, optimizer, train_dataloader, val_dataloader, ck_epoch, CHECKPOINT_PATH, args, writer)