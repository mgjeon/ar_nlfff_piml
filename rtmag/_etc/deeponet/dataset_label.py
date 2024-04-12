import numpy as np
from pathlib import Path

import torch
from torch.utils.data import Dataset

class DeepONetDatasetCNNlabeldata(Dataset):

    def __init__(self, file_list, b_norm, spatial_norm, cube_shape, 
                 bottom_batch_coords=1,
                 data_batch_coords=1, 
                 random_batch_coords=1):
        super().__init__()
        self.cube_shape = np.array([[0, cube_shape[0] - 1], [0, cube_shape[1] - 1], [0, cube_shape[2] - 1]])
        self.files = file_list
        self.b_norm = b_norm
        self.spatial_norm = spatial_norm
        self.bottom_batch_coords = int(bottom_batch_coords)
        self.data_batch_coords = int(data_batch_coords)
        self.random_batch_coords = int(random_batch_coords)
        self.float_tensor = torch.FloatTensor
        self.coords_shape = cube_shape

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        input_file = Path(self.files[idx])
        label_file = input_file.parent.parent / 'label' / input_file.name.replace('input_', 'label_')

        inputs = np.load(input_file, mmap_mode='r')
        labels = np.load(label_file, mmap_mode='r')['label'][:, :-1, :-1, :-1]

        branch_input = inputs['input'][:, :-1, :-1, :][..., 0].astype(np.float32)
        branch_input = branch_input / self.b_norm

        #---------------------------------------------------------
        data_values = labels.transpose(1, 2, 3, 0)
        b_shape = data_values.shape
        data_values = data_values.reshape(-1, 3).astype(np.float32)
        data_values = data_values / self.b_norm        

        data_coords = np.stack(np.mgrid[:b_shape[0], :b_shape[1], :b_shape[2]], -1).astype(np.float32)
        data_coords = data_coords.reshape(-1, 3).astype(np.float32)
        data_coords = data_coords / self.spatial_norm

        #---------------------------------------------------------
        b_slices = inputs['input'][:, :-1, :-1, :].transpose(1, 2, 3, 0)
        bottom_values = b_slices.reshape(-1, 3).astype(np.float32)
        bottom_values = bottom_values / self.b_norm        

        bottom_coords = np.stack(np.mgrid[:b_slices.shape[0], :b_slices.shape[1], :b_slices.shape[2]], -1).reshape(-1, 3).astype(np.float32)
        bottom_coords = bottom_coords / self.spatial_norm

        #---------------------------------------------------------
        random_coords = self.float_tensor(self.random_batch_coords, 3).uniform_()
        random_coords[:, 0] = (random_coords[:, 0] * (self.cube_shape[0, 1] - self.cube_shape[0, 0]) + self.cube_shape[0, 0])
        random_coords[:, 1] = (random_coords[:, 1] * (self.cube_shape[1, 1] - self.cube_shape[1, 0]) + self.cube_shape[1, 0])
        random_coords[:, 2] = (random_coords[:, 2] * (self.cube_shape[2, 1] - self.cube_shape[2, 0]) + self.cube_shape[2, 0])
        random_coords = random_coords / self.spatial_norm

        #--- pick bottom points
        r = np.random.choice(bottom_coords.shape[0], self.bottom_batch_coords)
        bottom_coords = bottom_coords[r]
        bottom_values = bottom_values[r]

        #--- pick slice points
        r = np.random.choice(data_coords.shape[0], self.data_batch_coords)
        data_coords = data_coords[r]
        data_values = data_values[r]

        
        samples = {'branch_input': branch_input,
                   'random_coords': random_coords,
                   'bottom_coords': bottom_coords,
                   'bottom_values': bottom_values,
                   'data_coords': data_coords,
                   'data_values': data_values}

        return samples