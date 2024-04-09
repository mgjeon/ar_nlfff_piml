import gc
import time
import pickle
import torch
import argparse
import json
import pandas as pd
from pathlib import Path
import numpy as np
from rtmag.paper.metric import evaluate_energy
from rtmag.paper.load import load_input_label, MyModel
from rtmag.paper.parse import parse_tai_string

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
input_files, label_files = load_input_label(configs.data_path)
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
for i in range(idx, len(input_files)):
    start_time = time.time()

    if isinstance(configs.idx, int):
        if i != configs.idx:
            continue

    with torch.no_grad():
        #----------------------------------------------------------------------------------
        input_file  = input_files[i]
        b = mm.get_pred(input_file)
        
        #----------------------------------------------------------------------------------
        label_file = label_files[i]
        assert input_file.stem[12:] == label_file.stem[12:]
        B = mm.get_label(label_file)
        Bp = mm.get_pot(label_file)

        #----------------------------------------------------------------------------------
        tstr = input_file.name[12:-4]
        obstime = parse_tai_string(tstr)
        res = {}
        res['obstime'] = obstime
        dx, dy, dz, dV = mm.get_dV(input_file)
        res.update(evaluate_energy(b, B, Bp, dV))
        results.append(res)
        df = pd.DataFrame.from_dict(results)
        csv_path = result_path / 'real_time.csv'
        df.to_csv(csv_path, index=False)

        with open(res_path, 'wb') as f:
            pickle.dump(results, f)
        
        with open(idx_path, 'wb') as f:
            pickle.dump(i+1, f)

        del b
        del B
        del Bp
        
        torch.cuda.empty_cache()
        gc.collect()

        time_interval = time.time() - start_time
        times.append(time_interval)
        with open(tim_path, 'wb') as f:
            pickle.dump(times, f)

        _, _, _, dVV = mm.get_dV(input_file)

        with open(result_path / "log.txt", "a") as f:
            f.write(f"{obstime} | {time_interval:.2f} s")
            print(f"{obstime} | {time_interval:.2f} s")
        
with open(result_path / "log.txt", "a") as f:
    f.write(f"Total Time: {np.sum(times)} seconds\n")
    print(f"Total Time: {np.sum(times)} seconds")