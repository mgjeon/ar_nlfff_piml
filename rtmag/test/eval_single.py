import numpy as np
import matplotlib.pyplot as plt

def dot_product(a, b):
    """
    a : [Nx, Ny, Nz, 3]
    b : [Nx, Ny, Nz, 3]
    """
    return (a * b).sum(-1)

def vector_norm(F):
    """
    F : [Nx, Ny, Nz, 3]
    """
    return (F**2).sum(-1)**0.5

def divergence(b_field):  # (x, y, z, (xyz-field))
    div_B = np.stack([np.gradient(b_field[..., i], axis=i, edge_order=2) for i in range(3)], axis=-1).sum(-1)
    return div_B


def curl(b_field):  # (x, y, z)
    _, dFx_dy, dFx_dz = np.gradient(b_field[..., 0], axis=[0, 1, 2], edge_order=2)
    dFy_dx, _, dFy_dz = np.gradient(b_field[..., 1], axis=[0, 1, 2], edge_order=2)
    dFz_dx, dFz_dy, _ = np.gradient(b_field[..., 2], axis=[0, 1, 2], edge_order=2)

    rot_x = dFz_dy - dFy_dz
    rot_y = dFx_dz - dFz_dx
    rot_z = dFy_dx - dFx_dy

    return np.stack([rot_x, rot_y, rot_z], -1)


def lorentz_force(b_field, j_field=None):
    j_field = j_field if j_field is not None else curl(b_field)
    l = np.cross(j_field, b_field, axis=-1)
    return l


def vector_norm(vector):
    return np.sqrt((vector ** 2).sum(-1))

def energy(b):
    return (b ** 2).sum(-1) / (8 * np.pi)

def angle(b_field, j_field=None):
    j_field = j_field if j_field is not None else curl(b_field)
    norm = vector_norm(b_field) * vector_norm(j_field) + 1e-7
    j_cross_b = np.cross(j_field, b_field, axis=-1)
    sig = vector_norm(j_cross_b) / norm
    return np.arcsin(np.clip(sig, -1. + 1e-7, 1. - 1e-7)) * (180 / np.pi)


def normalized_divergence(b_field):
    return np.abs(divergence(b_field)) / (vector_norm(b_field) + 1e-7)


def weighted_theta(b, j=None):
    j = j if j is not None else curl(b)
    sigma = vector_norm(lorentz_force(b, j)) / vector_norm(b) / vector_norm(j)
    w_sigma = np.average((sigma), weights=vector_norm(j))
    theta_j = np.arcsin(w_sigma) * (180 / np.pi)
    return theta_j


def mse(pred, true):
    return np.mean((pred - true) ** 2)

def relative_mse(pred, true):
    return mse(pred, true) / mse(true, np.zeros_like(true))

def relative_l2_error(pred, true):
    return np.linalg.norm(pred - true) / np.linalg.norm(true)

def C_vec(b, B):
    return dot_product(B, b).sum() / np.sqrt((vector_norm(B)**2).sum() * (vector_norm(b)**2).sum())

def C_cs(b, B):
    nu = dot_product(B, b)
    de = vector_norm(B) * vector_norm(b)
    M = np.sum([de!=0.])
    return (1 / M) * np.divide(nu, de, where=de!=0.).sum()

def En_prime(b, B):
    return 1 - (vector_norm(b - B).sum() / vector_norm(B).sum())

def Em_prime(b, B):
    nu = vector_norm(b - B)
    de = vector_norm(B)
    M = np.sum([de!=0.])
    return 1 - ((1 / M) * np.sum(np.divide(nu, de, where=de!=0.)))

def eps(b, B):
    return (vector_norm(b) ** 2).sum() / (vector_norm(B) ** 2).sum()

def Inspector(target, fake, eps=1e-7, ccc=True):
            
        rd = target - np.mean(target)
        fd = fake - np.mean(fake)
        
        r_num = np.sum(rd * fd)
        r_den = np.sqrt(np.sum(rd ** 2)) * np.sqrt(np.sum(fd ** 2))
        PCC_val = r_num/(r_den + eps)
        
        #----------------------------------------------------------------------
        if ccc == True:
            numerator = 2*PCC_val*np.std(target)*np.std(fake)
            denominator = (np.var(target) + np.var(fake)
                            + (np.mean(target) - np.mean(fake))**2)
            
            CCC_val = numerator/(denominator + eps)
            loss_CC = (1.0 - CCC_val)
        
        else:
            loss_CC = (1.0 - PCC_val)
            
        #----------------------------------------------------------------------
        return loss_CC


def evaluate(b, B):
    result = {}
    result['c_vec'] = np.sum((B * b).sum(-1)) / np.sqrt((B ** 2).sum(-1).sum() * (b ** 2).sum(-1).sum())
    M = np.prod(B.shape[:-1])
    result['c_cs'] = 1 / M * np.sum((B * b).sum(-1) / vector_norm(B) / vector_norm(b))

    result['E_n'] = 1 - ( vector_norm(b - B).sum() / vector_norm(B).sum())

    result['E_m'] = 1 - (1 / M * (vector_norm(b - B) / vector_norm(B)).sum())

    result['eps'] = (vector_norm(b) ** 2).sum() / (vector_norm(B) ** 2).sum()

    # B_potential = get_potential_field(B[:, :, 0, 2], 64)
    #
    # result['eps_p'] = (vector_norm(b[:, :, :64]) ** 2).sum() / (vector_norm(B_potential) ** 2).sum()
    # result['eps_p_ll'] = (vector_norm(B[:, :, :64]) ** 2).sum() / (vector_norm(B_potential) ** 2).sum()

    return result

def sig_J(b, j):
    return (vector_norm(np.cross(j, b, -1)) / vector_norm(b)).sum() / (vector_norm(j).sum() + 1e-6) * 1e2

def L1(b, j):
    return (vector_norm(np.cross(j, b, -1)) ** 2 / vector_norm(b) ** 2).mean()

def L2(b):
    return (divergence(b) ** 2).mean()

def L2n(b):
    return (np.abs(divergence(b)) / (vector_norm(b) + 1e-8)).mean() * 1e2

def angles(b, j):
    return angle(b, j).mean()

def normalized_divergences(b):
    return normalized_divergence(b).mean()

def self_eval(b, suffix=""):
    result = {}

    j = curl(b)
    result[f'sig_J{suffix}'] = sig_J(b, j)

    result[f'L1{suffix}'] = L1(b, j)

    result[f'L2{suffix}'] = L2(b)

    result[f'L2n{suffix}'] = L2n(b)

    result[f'angle{suffix}'] = angles(b, j)

    result[f'theta_w{suffix}'] = weighted_theta(b, j)

    result[f'norm_div{suffix}'] = normalized_divergences(b)
    return result

