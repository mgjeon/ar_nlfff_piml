import torch 
from torch import nn
import numpy as np
from tqdm import tqdm

def get_b(res, device):
    checkpoint = torch.load(res, map_location=device)
    model = nn.DataParallel(checkpoint['model']).to(device)

    cube_shape = checkpoint['cube_shape']
    b_norm = checkpoint['b_norm']
    spatial_norm = checkpoint['spatial_norm']

    nx, ny, nz = cube_shape
    coords = np.stack(np.mgrid[:nx, :ny, :nz], -1).astype(np.float32)
    coords_shape = coords.shape
    coords = coords.reshape(-1, 3)
    coords = torch.tensor(coords, dtype=torch.float32)
    coords = coords / spatial_norm

    coords = coords.view((-1, 3))
    cube = []
    batch_size = 10000
    it = range(int(np.ceil(coords.shape[0] / batch_size)))
    it = tqdm(it)
    for k in it:
        coord = coords[k * batch_size: (k + 1) * batch_size]
        coord = coord.to(device)
        coord.requires_grad = True
        cube += [model(coord).detach().cpu()]

    cube = torch.cat(cube)
    cube = cube.view(*coords_shape).numpy()
    b = cube * b_norm
    return b