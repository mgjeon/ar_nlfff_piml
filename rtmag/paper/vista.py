import numpy as np
import pyvista as pv
from rtmag.paper.metric import current_density, curl

def create_coordinates(bounds):
    xbounds = (bounds[0], bounds[1])
    ybounds = (bounds[2], bounds[3])
    zbounds = (bounds[4], bounds[5])
    meshgrid = np.mgrid[xbounds[0]:xbounds[1]+1, ybounds[0]:ybounds[1]+1, zbounds[0]:zbounds[1]+1]
    return np.stack(meshgrid, axis=-1).astype(np.float32)

def create_mesh(b, x=None, y=None, z=None):
    bx, by, bz = b[..., 0], b[..., 1], b[..., 2]
    bx, by, bz = map(np.array, (bx, by, bz))

    if (x is not None) and (y is not None) and (z is not None):
        mesh = pv.StructuredGrid(x, y, z)







    Nx, Ny, Nz = bx.shape
    co_bounds = (0, Nx-1, 0, Ny-1, 0, Nz-1)
    co_coords = create_coordinates(co_bounds).reshape(-1, 3)
    co_coord = co_coords.reshape(Nx, Ny, Nz, 3)
    x = co_coord[..., 0]
    y = co_coord[..., 1]
    z = co_coord[..., 2]
    mesh = pv.StructuredGrid(x, y, z)
    vectors = np.stack([bx, by, bz], axis=-1).transpose(2, 1, 0, 3).reshape(-1, 3)
    mesh['vector'] = vectors
    mesh.active_vectors_name = 'vector'
    magnitude = np.linalg.norm(vectors, axis=-1)
    mesh['magnitude'] = magnitude
    mesh.active_scalars_name = 'magnitude'

    j = curl(b)
    j_vec = np.stack([j[..., 0], j[..., 1], j[..., 2]], axis=-1).transpose(2, 1, 0, 3).reshape(-1, 3)
    mesh['current'] = j_vec
    j_mag = np.linalg.norm(j_vec, axis=-1)
    mesh['current_magnitude'] = j_mag

    if (dx is not None) and (dy is not None) and (dz is not None):
        j = current_density(b, dx, dy, dz)
        j_vec = np.stack([j[..., 0], j[..., 1], j[..., 2]], axis=-1).transpose(2, 1, 0, 3).reshape(-1, 3)
        mesh['current_density'] = j_vec
        j_mag = np.linalg.norm(j_vec, axis=-1)
        mesh['current_density_magnitude'] = j_mag

    return mesh