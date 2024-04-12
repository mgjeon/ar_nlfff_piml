import os 
import glob
import gc
import time
import pickle
import torch
import argparse
import json
import pandas as pd
from pathlib import Path
import numpy as np
from rtmag.paper.metric import evaluate_sharp
from rtmag.paper.load import MyModel
from rtmag.paper.parse import parse_tai_string
from sunpy.map import Map
from skimage.transform import resize

#-----------------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str)
configs = parser.parse_args()

with open(configs.config) as config:
    info = json.load(config)
    for key, value in info.items():
        configs.__dict__[key] = value

result_path = Path(configs.result_path)
result_path.mkdir(exist_ok=True, parents=True)

#-----------------------------------------------------------------------------------------
data_path = configs.sharp_data_path
hmi_p_files = sorted(glob.glob(os.path.join(data_path, '*Bp.fits')))  # x
hmi_t_files = sorted(glob.glob(os.path.join(data_path, '*Bt.fits')))  # y
hmi_r_files = sorted(glob.glob(os.path.join(data_path, '*Br.fits')))  # z
data_paths = list(zip(hmi_p_files, hmi_t_files, hmi_r_files))
clip = configs.clip
mm = MyModel(configs.meta_path, clip=clip)

#-----------------------------------------------------------------------------------------

results = []
idx = 0
times = []

res_path = result_path / 'real_time.pkl'
idx_path = result_path / 'real_time_idx.pkl'
tim_path = result_path / 'real_time_tim.pkl'

if res_path.exists():
    with open(res_path, 'rb') as f:
        results = pickle.load(f)

    with open(idx_path, 'rb') as f:
        idx = pickle.load(f)

    with open(tim_path, 'rb') as f:
        times = pickle.load(f)


#-----------------------------------------------------------------------------------------
for i in range(idx, len(data_paths)):
    start_time = time.time()

    if isinstance(configs.idx, int):
        if i != configs.idx:
            continue

    with torch.no_grad():
        #----------------------------------------------------------------------------------
        data_path = data_paths[i]
        hmi_p, hmi_t, hmi_r, = data_path
        p_map, t_map, r_map = Map(hmi_p), Map(hmi_t), Map(hmi_r)
        maps = [p_map, t_map, r_map]
        hmi_data = np.stack([maps[0].data, -maps[1].data, maps[2].data]).transpose()
        hmi_data = np.nan_to_num(hmi_data, nan=0.0)
        hmi_data = hmi_data.astype(np.float32)

        ox, oy, _ = hmi_data.shape
        nx, ny = 512, 256

        l = 0.36 # Mm

        dx = (ox * l)/nx
        dy = (oy * l)/ny
        dz = dy 

        dx, dy, dz = dx * 1e8, dy * 1e8, dz * 1e8  # cm
        dV = dx * dy * dz # cm^3
        model_input = resize(hmi_data, (nx, ny, 3))
        model_input = model_input[None, :, :, None, :]
        model_input = model_input.transpose(0, 3, 2, 1, 4)

        #----------------------------------------------------------------------------------
        b = mm.get_pred_from_numpy(model_input)

        #----------------------------------------------------------------------------------
        tstr = Path(data_path[0]).name[-27:-12]
        obstime = parse_tai_string(tstr)
        res = {}
        res['obstime'] = obstime
        res.update(evaluate_sharp(b, dV))
        results.append(res)
        df = pd.DataFrame.from_dict(results)
        csv_path = result_path / 'real_time.csv'
        df.to_csv(csv_path, index=False)

        with open(res_path, 'wb') as f:
            pickle.dump(results, f)
        
        with open(idx_path, 'wb') as f:
            pickle.dump(i+1, f)

        del b
        
        torch.cuda.empty_cache()
        gc.collect()

        time_interval = time.time() - start_time
        times.append(time_interval)
        with open(tim_path, 'wb') as f:
            pickle.dump(times, f)

        with open(result_path / "log.txt", "a") as f:
            f.write(f"{obstime} | {time_interval:.2f} s")
            print(f"{obstime} | {time_interval:.2f} s")
        
with open(result_path / "log.txt", "a") as f:
    f.write(f"Total Time: {np.sum(times)} seconds\n")
    print(f"Total Time: {np.sum(times)} seconds")