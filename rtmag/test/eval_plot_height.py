import numpy as np
import matplotlib.pyplot as plt

def evals(func, name, b, B_pot, B_lab, lab=1e-7):
    heights = np.arange(b.shape[2])

    plots_b = []
    for i in range(b.shape[-2]):
        plots_b.append(func(b[:, :, i, :], B_lab[:, :, i, :]))

    plots_B_pot = []
    for i in range(B_pot.shape[-2]):
        plots_B_pot.append(func(B_pot[:, :, i, :], B_lab[:, :, i, :]))

    plots_B_lab = []
    for i in range(B_lab.shape[-2]):
        plots_B_lab.append(func(B_lab[:, :, i, :], B_lab[:, :, i, :]) + lab)

    fig = plt.figure(figsize=(6, 8))
    plt.plot(plots_b, heights, color='red', label='PINO')
    plt.plot(plots_B_pot, heights, color='black', label='Potential')
    plt.plot(plots_B_lab, heights, color='blue', label='Ground Truth')
    plt.legend()
    plt.xlabel(name)
    plt.ylabel('height [pixel]')
    plt.xscale('log')
    plt.yscale('linear')
    plt.grid()


def j_evals(func, name, b, B_pot, B_lab, j, J_pot, J_lab):
    heights = np.arange(b.shape[2])

    plots_b = []
    for i in range(b.shape[-2]):
        plots_b.append(func(b[:, :, i, :], j[:, :, i, :]))

    plots_B_pot = []
    for i in range(B_pot.shape[-2]):
        plots_B_pot.append(func(B_pot[:, :, i, :], J_pot[:, :, i, :]))

    plots_B_lab = []
    for i in range(B_lab.shape[-2]):
        plots_B_lab.append(func(B_lab[:, :, i, :], J_lab[:, :, i, :]))

    fig = plt.figure(figsize=(6, 8))
    plt.plot(plots_b, heights, color='red', label='PINO')
    plt.plot(plots_B_pot, heights, color='black', label='Potential')
    plt.plot(plots_B_lab, heights, color='blue', label='Ground Truth')
    plt.legend()
    plt.xlabel(name)
    plt.ylabel('height [pixel]')
    plt.xscale('log')
    plt.yscale('linear')
    plt.grid()


def self_evals(func, name, b, B_pot, B_lab):
    plots_b = []
    for i in range(0, b.shape[-2]-3):
        plots_b.append(func(b[:, :, i:i+3, :]))

    plots_B_pot = []
    for i in range(B_pot.shape[-2]-3):
        plots_B_pot.append(func(B_pot[:, :, i:i+3, :]))

    plots_B_lab = []
    for i in range(B_lab.shape[-2]-3):
        plots_B_lab.append(func(B_lab[:, :, i:i+3, :]))

    heightss = np.arange(b.shape[2]-3)
    fig = plt.figure(figsize=(6, 8))
    plt.plot(plots_b, heightss, color='red', label='PINO')
    plt.plot(plots_B_pot, heightss, color='black', label='Potential')
    plt.plot(plots_B_lab, heightss, color='blue', label='Ground Truth')
    plt.legend()
    plt.xlabel(name)
    plt.ylabel('height [pixel]')
    plt.xscale('log')
    plt.yscale('linear')
    plt.grid()