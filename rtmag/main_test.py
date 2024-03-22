import gc
import json
import time
import pickle
import argparse
from pathlib import Path
from datetime import datetime
import shutil

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyvista as pv
pv.start_xvfb()
pv.set_jupyter_backend('static')

from neuralop.models import UNO

from rtmag.test.field_plot import create_mesh, mag_plotter
from rtmag.test.pre import parse_tai_string, plot_sample, plot_lines, plot_loss, evaluate_bp, plot_loss_2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#-----------------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str)
configs = parser.parse_args()

with open(configs.config) as config:
    info = json.load(config)
    for key, value in info.items():
        configs.__dict__[key] = value

result_path = Path(configs.result_path)
if configs.overwrite:
    shutil.rmtree(str(result_path), ignore_errors=True)
result_path.mkdir(exist_ok=True, parents=True)

input_path = Path(configs.input_path).glob('*.npz')
input_path = sorted(input_path)

label_path = Path(configs.label_path).glob('*.npz')
label_path = sorted(label_path)


#-----------------------------------------------------------------------------------------
meta_path = Path(configs.meta_path)
checkpoint = torch.load(meta_path / "best_model.pt", map_location=device)

args = argparse.Namespace()
info = np.load(meta_path / 'args.npy', allow_pickle=True).item()
for key, value in info.items():
        args.__dict__[key] = value

isNone = False
b_norm = args.data["b_norm"]
if b_norm is None:
    isNone = True

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


#-----------------------------------------------------------------------------------------
start_time = time.time()

results = []
idx = 0

res_path = result_path / 'real_time.pkl'
idx_path = result_path / 'real_time_idx.pkl'

if res_path.exists():
    with open(res_path, 'rb') as f:
        results = pickle.load(f)

    with open(idx_path, 'rb') as f:
        idx = pickle.load(f)


for i in range(idx, len(input_path)):

    if isinstance(configs.idx, int):
        if i != configs.idx:
            continue

    with torch.no_grad():
        #----------------------------------------------------------------------------------
        input_file  = input_path[i]
        inputs = np.load(input_file)
        model_input = torch.from_numpy(inputs['input'])[None, ...] 
        model_input = model_input[:, :, :-1, :-1, :]  # remove duplicated periodic boundary
        
        if isNone:
            b_norm = torch.max(torch.abs(model_input)).item()

        model_input /= b_norm
        # [batch_size, 3, 513, 257, 1]
        model_input = model_input.to(device)
        model_input = torch.permute(model_input, (0, 4, 3, 2, 1))

        # input : [batch_size,   1, 256, 512, 3]
        # output: [batch_size, 256, 256, 512, 3]
        model_output = model(model_input)

        # [512, 256, 256, 3]
        b = model_output.detach().cpu().numpy().transpose(0, 3, 2, 1, 4)[0]
        divi = (b_norm / np.arange(1, b.shape[2] + 1)).reshape(1, 1, -1, 1)
        b = b * divi
        
        #----------------------------------------------------------------------------------
        # B, Bp
        label_file = label_path[i]
        B = np.load(label_file)['label'].transpose(1, 2, 3, 0).astype(np.float32)
        Bp = np.load(label_file)['pot'].transpose(1, 2, 3, 0).astype(np.float32)

        # remove duplicated periodic boundary
        B = B[:-1, :-1, :-1, :]
        Bp = Bp[:-1, :-1, :-1, :]

        if configs.metrics.get("z_max", False):
            b = b[:, :, :configs.metrics["z_max"], :]
            B = B[:, :, :configs.metrics["z_max"], :]
            Bp = Bp[:, :, :configs.metrics["z_max"], :]

        # dV
        dx, dy, dz = inputs['dx'], inputs['dy'], inputs['dz']  # Mm
        dx, dy, dz = dx * 1e8, dy * 1e8, dz * 1e8  # cm
        dV = dx * dy * dz # cm^3

        #----------------------------------------------------------------------------------
        tstr = input_file.name[12:-4]
        obstime = parse_tai_string(tstr)
        res = {}
        res['obstime'] = obstime
        res.update(evaluate_bp(b, B, Bp, dV))
        print(f"{obstime}|b_norm{b_norm:.0f}|C_vec:{res['C_vec']:.2f}|C_cs:{res['C_cs']:.2f}|E_n':{res['E_n_prime']:.2f}|E_m':{res['E_m_prime']:.2f}|eps:{res['eps']:.2f}")
        results.append(res)
        res["b_norm"] = b_norm
        df = pd.DataFrame.from_dict(results)
        csv_path = result_path / 'real_time.csv'
        df.to_csv(csv_path, index=False)

        #----------------------------------------------------------------------------------
        plot_path = result_path / "plot" 
        plot_path.mkdir(parents=True, exist_ok=True)
        
        line_path = result_path / "line" 
        line_path.mkdir(parents=True, exist_ok=True)

        loss_path = result_path / "loss"
        loss_path.mkdir(parents=True, exist_ok=True)

        plot_sample(plot_path / f"{tstr}.png", b, B, f"PINO vs ISEE {tstr}")
        plot_lines(b, B, f'PINO {tstr}', f'ISEE {tstr}', line_path / f"{tstr}.png")
        # plot_loss(loss_path / f"{tstr}.png", b, B, Bp, f"PINO vs ISEE {tstr}", height=b.shape[2])
        plot_loss_2(b, B, Bp, loss_path / f"{tstr}_rel_l2_err.png", loss_path / f"{tstr}_eps.png")
        
        with open(res_path, 'wb') as f:
            pickle.dump(results, f)
        
        with open(idx_path, 'wb') as f:
            pickle.dump(i+1, f)

        del b, B, dV, dx, dy, dz, inputs, model_input, model_output, res

        torch.cuda.empty_cache()
        gc.collect()

time_interval = time.time() - start_time

print(f"--- {time_interval} seconds ---")

with open(result_path / "log.txt", "a") as f:
    f.write(f"--- {time_interval} seconds --- {datetime.now()}")