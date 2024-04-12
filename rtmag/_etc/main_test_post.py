import pandas as pd
import json
import argparse
from pathlib import Path

from rtmag.test.post import Metrics

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str)
configs = parser.parse_args()

with open(configs.config) as config:
    info = json.load(config)
    for key, value in info.items():
        configs.__dict__[key] = value


result_path = Path(configs.result_path)

result_csv = result_path / "real_time.csv"

f = pd.read_csv(result_csv)
m = Metrics(f,
            title=configs.metrics["title"],
            energy_ylim=configs.metrics["energy_ylim"],
            free_energy_ylim=configs.metrics["free_energy_ylim"],
            metrics_ylim=configs.metrics["metrics_ylim"],
            error_ylim=configs.metrics["error_ylim"],
            eps_ylim=configs.metrics["eps_ylim"],
            hour=configs.metrics["hour"])
m.print()

with open(result_path / "metrics.txt", "w") as f:
    df = m.df
    eps_mean = m.eps.mean()
    c_vec_mean = df['C_vec'].mean()
    c_cs_mean = df['C_cs'].mean()
    e_n_mean = df["E_n_prime"].mean()
    e_m_mean = df["E_m_prime"].mean()
    l2_err_mean = df['l2_err'].mean()
    f.write(f"avg C_vec        : {c_vec_mean:.2f}\n")
    f.write(f"avg C_cs         : {c_cs_mean:.2f}\n")
    f.write(f"avg E_n_prime    : {e_n_mean:.2f}\n")
    f.write(f"avg E_m_prime    : {e_m_mean:.2f}\n")
    f.write(f"avg eps          : {eps_mean:.2f}\n")
    f.write(f"avg l2_err       : {l2_err_mean:.2f}\n")

m.plot(result_path)