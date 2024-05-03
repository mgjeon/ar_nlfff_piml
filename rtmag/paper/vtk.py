import numpy as np
from tvtk.api import tvtk, write_data
import os

def save_vtk(vec, path, name, scalar=None, scalar_name='scalar', Mm_per_pix=720e-3):
    """Save numpy array as VTK file

    :param vec: numpy array of the vector field (x, y, z, c)
    :param path: path to the target VTK file
    :param name: label of the vector field (e.g., B)
    :param Mm_per_pix: pixel size in Mm. 360e-3 for original HMI resolution. (default bin2 pixel scale)
    """
    # Unpack
    dim = vec.shape[:-1]
    # Generate the grid
    pts = np.stack(np.mgrid[0:dim[0], 0:dim[1], 0:dim[2]], -1).astype(np.int64) * Mm_per_pix
    # reorder the points and vectors in agreement with VTK
    # requirement of x first, y next and z last.
    pts = pts.transpose(2, 1, 0, 3)
    pts = pts.reshape((-1, 3))
    vectors = vec.transpose(2, 1, 0, 3)
    vectors = vectors.reshape((-1, 3))

    sg = tvtk.StructuredGrid(dimensions=dim, points=pts)
    sg.point_data.vectors = vectors
    sg.point_data.vectors.name = name
    if scalar is not None:
        scalars = scalar.transpose(2, 1, 0)
        scalars = scalars.reshape((-1))
        sg.point_data.add_array(scalars)
        sg.point_data.get_array(1).name = scalar_name
        sg.point_data.update()

    write_data(sg, path)

import pyvista as pv
from rtmag.paper.metric import current_density, curl

def save_vtk_xyz(b, x, y, z, path, overwrite=False):
    """
    b : [Nx, Ny, Nz, 3]
    x : [Nx]  Mm
    y : [Ny]  Mm
    z : [Nz]  Mm

    x, y, z = np.meshgrid(x1, y1, z1, indexing='ij')
    """

    if not overwrite:
        if os.path.exists(path):
            print(f'{path} already exists')
            return

    dx = x[1] - x[0]
    dy = y[1] - y[0]
    dz = z[1] - z[0]

    x, y, z = np.meshgrid(x, y, z, indexing='ij')
    mesh = pv.StructuredGrid(x, y, z)

    bx, by, bz = b[..., 0], b[..., 1], b[..., 2]
    vectors = np.stack([bx, by, bz], axis=-1).transpose(2, 1, 0, 3).reshape(-1, 3)
    mesh['vector'] = vectors
    mesh.active_vectors_name = 'vector'

    magnitude = np.linalg.norm(vectors, axis=-1)
    mesh['magnitude'] = magnitude
    mesh.active_scalars_name = 'magnitude'

    j = curl(b)
    j_vec = np.stack([j[..., 0], j[..., 1], j[..., 2]], axis=-1).transpose(2, 1, 0, 3).reshape(-1, 3)
    mesh['current_density_per_pixel'] = j_vec
    j_mag = np.linalg.norm(j_vec, axis=-1)
    mesh['current_density_per_pixel_magnitude'] = j_mag

    
    dx = dx * 1e8 # cm
    dy = dy * 1e8 # cm
    dz = dz * 1e8 # cm
    j = current_density(b, dx, dy, dz)
    j_vec = np.stack([j[..., 0], j[..., 1], j[..., 2]], axis=-1).transpose(2, 1, 0, 3).reshape(-1, 3)
    mesh['current_density'] = j_vec
    j_mag = np.linalg.norm(j_vec, axis=-1)
    mesh['current_density_magnitude'] = j_mag

    mesh.save(path)