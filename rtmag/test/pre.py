from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyvista as pv
pv.start_xvfb()

from rtmag.test.field_plot import create_mesh, mag_plotter
from rtmag.test.eval import C_vec, C_cs, En_prime, Em_prime, eps, l2_error, energy
from rtmag.test.eval_single import self_eval


def parse_tai_string(tstr):
    year   = int(tstr[:4])
    month  = int(tstr[4:6])
    day    = int(tstr[6:8])
    hour   = int(tstr[9:11])
    minute = int(tstr[11:13])
    return datetime(year, month, day, hour, minute)

def plot_sample(save_path, b, B, suptitle, n_samples=5, v_mm=2500, show=False):

    fig, axs = plt.subplots(3*2, n_samples, figsize=(n_samples * 4, 12*2))
    heights = np.linspace(0, 1, n_samples) ** 2 * (b.shape[2] - 1)  # more samples from lower heights
    heights = heights.astype(np.int32)
    for i in range(3):
        for j, h in enumerate(heights):
            v_min_max = int(v_mm / (h+1))
            axs[2*i, j].imshow(b[:, :, h, i].transpose(), cmap='gray', vmin=-v_min_max, vmax=v_min_max,
                            origin='lower')
            axs[2*i, j].set_axis_off()
            if i == 0:
                axs[2*i, j].set_title(f'bx (z={h})', fontsize=20)
            elif i == 1:
                axs[2*i, j].set_title(f'by (z={h})', fontsize=20)
            elif i == 2:
                axs[2*i, j].set_title(f'bz (z={h})', fontsize=20)

    heights = np.linspace(0, 1, n_samples) ** 2 * (B.shape[2] - 1)  # more samples from lower heights
    heights = heights.astype(np.int32)
    for i in range(3):
        for j, h in enumerate(heights):
            v_min_max = int(v_mm / (h+1))
            axs[2*i+1, j].imshow(B[:, :, h, i].transpose(), cmap='gray', vmin=-v_min_max, vmax=v_min_max,
                            origin='lower')
            axs[2*i+1, j].set_axis_off()
            if i == 0:
                axs[2*i+1, j].set_title(f'Bx (z={h})', fontsize=20)
            elif i == 1:
                axs[2*i+1, j].set_title(f'By (z={h})', fontsize=20)
            elif i == 2:
                axs[2*i+1, j].set_title(f'Bz (z={h})', fontsize=20)
    
    fig.suptitle(suptitle, fontsize=30)
    plt.tight_layout()
    if show:
        plt.show()
    else:
        fig.savefig(save_path, dpi=300)
        plt.close("all")

# Relative Root Mean Squared Error (RRMSE)
def relative_root_mean_squared_error(true, pred):
    true = true.flatten()
    pred = pred.flatten()
    n = len(true) # update
    num = np.sum(np.square(true - pred)) / n  # update
    den = np.sum(np.square(pred))
    squared_error = num/den
    rrmse_loss = np.sqrt(squared_error)
    return rrmse_loss


def evaluate(b, B, dV):
    # b : model solution
    # B : reference magnetic field

    result = {}
    result["C_vec"] = C_vec(b, B)
    result["C_cs"] = C_cs(b, B)
    result["E_n'"] = En_prime(b, B)
    result["E_m'"] = Em_prime(b, B)
    result['eps'] = eps(b, B)
    result['l2_err'] = l2_error(b, B)
    result['RRMSE'] = relative_root_mean_squared_error(b, B)
    result['pred_E_1e33'] = energy(b, dV) / 1e33
    result['ref_E_1e33'] = energy(B, dV) / 1e33
    
    res = self_eval(b)
    result.update(res)

    # result["C_vec_nf2"] = np.sum((B * b).sum(-1)) / np.sqrt((B ** 2).sum(-1).sum() * (b ** 2).sum(-1).sum())
    # M = np.prod(B.shape[:-1])
    # result["C_cs_nf2"] = 1 / M * np.sum((B * b).sum(-1) / metric.vector_norm(B) / metric.vector_norm(b))
    # result["E_n'_nf2"] = metric.vector_norm(b - B).sum() / metric.vector_norm(B).sum()
    # result["E_m'_nf2"] = 1 / M * (metric.vector_norm(b - B) / metric.vector_norm(B)).sum()
    # result['eps_nf2'] = (metric.vector_norm(b) ** 2).sum() / (metric.vector_norm(B) ** 2).sum()
    # j = metric.curl(b)
    # J = metric.curl(B)
    # result["sig_J"] = (metric.vector_norm(np.cross(j, b, -1)) / metric.vector_norm(b)).sum() / (metric.vector_norm(j).sum() + 1e-6) * 1e2
    # result['sig_J_B'] = (metric.vector_norm(np.cross(J, B, -1)) / metric.vector_norm(B)).sum() / (metric.vector_norm(J).sum() + 1e-6) * 1e2
    # result['L1'] = (metric.vector_norm(np.cross(j, b, -1)) ** 2 / metric.vector_norm(b) ** 2).mean()
    # result['L2'] = (metric.divergence(b) ** 2).mean()
    # result['L1_B'] = (metric.vector_norm(np.cross(curl(B), B, -1)) ** 2 / metric.vector_norm(B) ** 2).mean()
    # result['L2_B'] = (metric.divergence(B) ** 2).mean()
    # result['L2n'] = (np.abs(metric.divergence(b)) / (metric.vector_norm(b) + 1e-8)).mean() * 1e2
    # result['L2n_B'] = (np.abs(metric.divergence(B)) / (metric.vector_norm(B) + 1e-8)).mean() * 1e2

    return result

def evaluate_bp(b, B, Bp, dV, self_eval=False):
    # b : model solution
    # B : reference magnetic field

    result = {}
    result["C_vec"] = C_vec(b, B)
    result["C_cs"] = C_cs(b, B)
    result["E_n_prime"] = En_prime(b, B)
    result["E_m_prime"] = Em_prime(b, B)
    result['eps'] = eps(b, B)
    result['l2_err'] = l2_error(b, B)
    result['RRMSE'] = relative_root_mean_squared_error(b, B)
    result['pred_E_1e33'] = energy(b, dV) / 1e33
    result['ref_E_1e33'] = energy(B, dV) / 1e33
    result['pot_E_1e33'] = energy(Bp, dV) / 1e33

    if self_eval is True:
        res = self_eval(b, "")
        result.update(res)

        res = self_eval(B, "_ref")
        result.update(res)

        res = self_eval(Bp, "_pot")
        result.update(res)

    return result

# def evaluate(b, B, dV_b, dV_B):
#     # b : model solution
#     # B : reference magnetic field

#     result = {}
#     result["C_vec"] = C_vec(b, B)
#     result["C_cs"] = C_cs(b, B)
#     result["E_n'"] = En_prime(b, B)
#     result["E_m'"] = Em_prime(b, B)
#     result['l2_err'] = l2_error(b, B)
#     result['pred_E_1e33'] = energy(b, dV_b) / 1e33
#     result['ref_E_1e33'] = energy(B, dV_B) / 1e33
#     result['eps'] = result['pred_E_1e33'] / result['ref_E_1e33']

#     return result

def eval_plots(b_pred, b_true, b_pot, func, name, savepath):
    heights = np.arange(b_pred.shape[2])

    plots_b = []
    for i in range(b_pred.shape[-2]):
        plots_b.append(func(b_pred[:, :, i, :], b_true[:, :, i, :]))

    plots_B_pot = []
    for i in range(b_pot.shape[-2]):
        plots_B_pot.append(func(b_pot[:, :, i, :], b_true[:, :, i, :]))

    fig = plt.figure(figsize=(6, 8))
    plt.plot(plots_b, heights, color='red', label='PINO')
    plt.plot(plots_B_pot, heights, color='black', label='Potential')
    plt.legend()
    plt.xlabel(name)
    plt.ylabel('height [pixel]')
    plt.xscale('log')
    plt.yscale('linear')
    plt.grid()
    plt.tight_layout()
    fig.savefig(savepath, dpi=300)
    return fig

def plot_loss_2(b_pred, b_true, b_pot, savepath_1, savepath_2):
    eval_plots(b_pred, b_true, b_pot, l2_error, "rel_l2_err", savepath_1)
    eval_plots(b_pred, b_true, b_pot, eps, "eps", savepath_2)

def plot_loss(save_path, b, B, Bp, title, height=257, show=False):
    def inspect_val(pred, target):

        pd = pred - pred.mean()
        td = target - target.mean()

        r_num = (pd * td).sum()
        r_den = np.sqrt((pd**2).sum()) * np.sqrt((td**2).sum())
        r = r_num / r_den

        return 1-r

    cc = []
    for i in range(height):
        cc.append(inspect_val(b[:, :, i, :], B[:, :, i, :]))

    mses = []
    for i in range(height):
        mses.append(l2_error(b[:, :, i, :], B[:, :, i, :]))

    fig = plt.figure(figsize=(5, 10))
    plt.subplot(2, 1, 1)
    plt.plot(mses)
    plt.ylabel('relative l2 error')
    plt.xlabel('z')
    # plt.ylim(0, 1000)
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.plot(cc)
    plt.ylabel('1-CC')
    # plt.ylim(-0.1, 1.1)
    plt.xlabel('z')
    plt.grid()


    fig.suptitle(title)
    plt.tight_layout()
    if show:
        plt.show()
    else:
        pd.DataFrame({'l2_err': mses, '1_cc': cc}).to_csv(save_path.parent / (save_path.stem + '.csv'), index=False)
        fig.savefig(save_path, dpi=300)
        plt.close("all")

def plot_lines(b, B, b_title, B_title, sv_path, vmin=-2500, vmax=2500, i_siz=160, j_siz=100, i_res=16, j_res=16, show=False, window_size=(1200, 800), zoom=2.0, max_time=100000):

    bx = b[..., 0]
    by = b[..., 1]
    bz = b[..., 2]

    i_siz = bx.shape[0] / 2
    j_siz = bx.shape[1] / 2
 
    mesh = create_mesh(bx, by, bz)
    b_plot = mag_plotter(mesh)
    b_tube, b_bottom, b_dargs = b_plot.create_mesh(i_siz=i_siz, j_siz=j_siz, i_resolution=i_res, j_resolution=j_res, vmin=vmin, vmax=vmax, max_time=max_time)

    B_x = B[..., 0]
    B_y = B[..., 1]
    B_z = B[..., 2]
    B_mesh = create_mesh(B_x, B_y, B_z)
    B_plot = mag_plotter(B_mesh)
    B_tube, B_bottom, B_dargs = B_plot.create_mesh(i_siz=i_siz, j_siz=j_siz, i_resolution=i_res, j_resolution=j_res, vmin=vmin, vmax=vmax, max_time=max_time)


    camera_position = 'xy'
    title_fontsize = 10
    # window_size = (1200, 800)

    if show:
        off_screen = False
    else:
        off_screen = True
    p = pv.Plotter(shape=(2, 1), off_screen=off_screen, window_size=window_size)
    p.subplot(0, 0)
    p.add_mesh(b_plot.grid.outline())
    p.add_mesh(b_bottom, cmap='gray', **b_dargs)
    p.add_mesh(b_tube, lighting=False, color='blue')
    p.camera_position = camera_position
    p.add_title(b_title, font_size=title_fontsize)
    p.camera.zoom(zoom)

    p.subplot(1, 0)
    p.add_mesh(B_plot.grid.outline())
    p.add_mesh(B_bottom, cmap='gray', **B_dargs)
    p.add_mesh(B_tube, lighting=False, color='blue')
    p.camera_position = camera_position
    p.add_title(B_title, font_size=title_fontsize)
    p.camera.zoom(zoom)

    if show:
        p.show()
    else:
        p.show(screenshot=sv_path)