import numpy as np
from pathlib import Path

import torch
from torch.utils.data import Dataset


class DeepONetBoundaryDataset(Dataset):

    def __init__(self, file_list, b_norm, spatial_norm):
        super().__init__()
        self.files = file_list
        self.b_norm = b_norm
        self.spatial_norm = spatial_norm

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        input_file = self.files[idx]
        inputs = np.load(input_file)
        b_slices = inputs['input'].transpose(1, 2, 3, 0)
        slices_coords = np.stack(np.mgrid[:b_slices.shape[0], :b_slices.shape[1], :b_slices.shape[2]], -1).astype(np.float32)

        slices_coords = slices_coords.reshape(-1, 3).astype(np.float32)
        slices_values = b_slices.reshape(-1, 3).astype(np.float32)

        #--- TODO: other boundary coords & values
        coords = slices_coords
        values = slices_values

        #--- normalize
        coords = coords / self.spatial_norm
        values = values / self.b_norm

        #--- DeepONet (branch input)
        bottom = values.flatten()

        #--- pick one data
        r = np.random.choice(coords.shape[0], 1)[0]
        coords = coords[r]
        values = values[r]
        
        samples = {'bottom': bottom,
                   'coords': coords,
                   'values': values}

        return samples
    

class DeepONetRandomDataset(Dataset):

    def __init__(self, file_list, b_norm, spatial_norm, cube_shape):
        super().__init__()
        self.cube_shape = np.array([[0, cube_shape[0] - 1], [0, cube_shape[1] - 1], [0, cube_shape[2] - 1]])
        self.files = file_list
        self.b_norm = b_norm
        self.spatial_norm = spatial_norm
        self.float_tensor = torch.FloatTensor

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        input_file = self.files[idx]
        inputs = np.load(input_file)
        b_slices = inputs['input'].transpose(1, 2, 3, 0)
        slices_values = b_slices.reshape(-1, 3).astype(np.float32)

        random_coords = self.float_tensor(1, 3).uniform_()
        random_coords[:, 0] = (random_coords[:, 0] * (self.cube_shape[0, 1] - self.cube_shape[0, 0]) + self.cube_shape[0, 0])
        random_coords[:, 1] = (random_coords[:, 1] * (self.cube_shape[1, 1] - self.cube_shape[1, 0]) + self.cube_shape[1, 0])
        random_coords[:, 2] = (random_coords[:, 2] * (self.cube_shape[2, 1] - self.cube_shape[2, 0]) + self.cube_shape[2, 0])

        #---
        coords = random_coords
        values = slices_values

        #--- normalize
        coords = coords / self.spatial_norm
        values = values / self.b_norm

        #--- DeepONet (branch input)
        bottom = values.flatten()

        #--- pick one data
        r = np.random.choice(coords.shape[0], 1)[0]
        coords = coords[r]
        
        samples = {'bottom': bottom,
                   'coords': coords}

        return samples
    

class DeepONetDataset(Dataset):

    def __init__(self, file_list, b_norm, spatial_norm, cube_shape, boundary_batch_coords=1, random_batch_coords=1):
        super().__init__()
        self.cube_shape = np.array([[0, cube_shape[0] - 1], [0, cube_shape[1] - 1], [0, cube_shape[2] - 1]])
        self.files = file_list
        self.b_norm = b_norm
        self.spatial_norm = spatial_norm
        self.boundary_batch_coords = int(boundary_batch_coords)
        self.random_batch_coords = int(random_batch_coords)
        self.float_tensor = torch.FloatTensor

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        input_file = self.files[idx]
        inputs = np.load(input_file)
        b_slices = inputs['input'].transpose(1, 2, 3, 0)

        slices_values = b_slices.reshape(-1, 3).astype(np.float32)
        slices_values = slices_values / self.b_norm
        #--- DeepONet (branch input)
        branch_input = slices_values.flatten()

        slices_coords = np.stack(np.mgrid[:b_slices.shape[0], :b_slices.shape[1], :b_slices.shape[2]], -1).astype(np.float32)
        slices_coords = slices_coords.reshape(-1, 3).astype(np.float32)
        slices_coords = slices_coords / self.spatial_norm

        random_coords = self.float_tensor(self.random_batch_coords, 3).uniform_()
        random_coords[:, 0] = (random_coords[:, 0] * (self.cube_shape[0, 1] - self.cube_shape[0, 0]) + self.cube_shape[0, 0])
        random_coords[:, 1] = (random_coords[:, 1] * (self.cube_shape[1, 1] - self.cube_shape[1, 0]) + self.cube_shape[1, 0])
        random_coords[:, 2] = (random_coords[:, 2] * (self.cube_shape[2, 1] - self.cube_shape[2, 0]) + self.cube_shape[2, 0])
        random_coords = random_coords / self.spatial_norm

        #--- pick one data
        r = np.random.choice(slices_coords.shape[0], self.boundary_batch_coords)
        slices_coords = slices_coords[r]
        slices_values = slices_values[r]
        
        samples = {'branch_input': branch_input,
                   'random_coords': random_coords,
                   'slices_coords': slices_coords,
                   'slices_values': slices_values}

        return samples
    

#-------------------------------------------------------------------------

class DeepONetDatasetCNN(Dataset):

    def __init__(self, file_list, b_norm, spatial_norm, cube_shape, boundary_batch_coords=1, random_batch_coords=1):
        super().__init__()
        self.cube_shape = np.array([[0, cube_shape[0] - 1], [0, cube_shape[1] - 1], [0, cube_shape[2] - 1]])
        self.files = file_list
        self.b_norm = b_norm
        self.spatial_norm = spatial_norm
        self.boundary_batch_coords = int(boundary_batch_coords)
        self.random_batch_coords = int(random_batch_coords)
        self.float_tensor = torch.FloatTensor

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        input_file = self.files[idx]
        inputs = np.load(input_file)

        branch_input = inputs['input'][..., 0].astype(np.float32)
        branch_input = branch_input / self.b_norm

        b_slices = inputs['input'].transpose(1, 2, 3, 0)
        slices_values = b_slices.reshape(-1, 3).astype(np.float32)
        slices_values = slices_values / self.b_norm        

        slices_coords = np.stack(np.mgrid[:b_slices.shape[0], :b_slices.shape[1], :b_slices.shape[2]], -1).astype(np.float32)
        slices_coords = slices_coords.reshape(-1, 3).astype(np.float32)
        slices_coords = slices_coords / self.spatial_norm

        random_coords = self.float_tensor(self.random_batch_coords, 3).uniform_()
        random_coords[:, 0] = (random_coords[:, 0] * (self.cube_shape[0, 1] - self.cube_shape[0, 0]) + self.cube_shape[0, 0])
        random_coords[:, 1] = (random_coords[:, 1] * (self.cube_shape[1, 1] - self.cube_shape[1, 0]) + self.cube_shape[1, 0])
        random_coords[:, 2] = (random_coords[:, 2] * (self.cube_shape[2, 1] - self.cube_shape[2, 0]) + self.cube_shape[2, 0])
        random_coords = random_coords / self.spatial_norm

        #--- pick one data
        r = np.random.choice(slices_coords.shape[0], self.boundary_batch_coords)
        slices_coords = slices_coords[r]
        slices_values = slices_values[r]
        
        samples = {'branch_input': branch_input,
                   'random_coords': random_coords,
                   'slices_coords': slices_coords,
                   'slices_values': slices_values}

        return samples
    
#-------------------------------------------------------------------------
    
class DeepONetDatasetCNN_Bnormalize(Dataset):

    def __init__(self, file_list, spatial_norm, cube_shape, boundary_batch_coords=1, random_batch_coords=1):
        super().__init__()
        self.cube_shape = np.array([[0, cube_shape[0] - 1], [0, cube_shape[1] - 1], [0, cube_shape[2] - 1]])
        self.files = file_list
        self.spatial_norm = spatial_norm
        self.boundary_batch_coords = int(boundary_batch_coords)
        self.random_batch_coords = int(random_batch_coords)
        self.float_tensor = torch.FloatTensor

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        input_file = self.files[idx]
        inputs = np.load(input_file)

        branch_input = inputs['input'][..., 0].astype(np.float32)
        b_norm = np.max(np.abs(branch_input))
        branch_input = branch_input / b_norm

        b_slices = inputs['input'].transpose(1, 2, 3, 0)
        slices_values = b_slices.reshape(-1, 3).astype(np.float32)
        slices_values = slices_values / b_norm      

        slices_coords = np.stack(np.mgrid[:b_slices.shape[0], :b_slices.shape[1], :b_slices.shape[2]], -1).astype(np.float32)
        slices_coords = slices_coords.reshape(-1, 3).astype(np.float32)
        slices_coords = slices_coords / self.spatial_norm

        random_coords = self.float_tensor(self.random_batch_coords, 3).uniform_()
        random_coords[:, 0] = (random_coords[:, 0] * (self.cube_shape[0, 1] - self.cube_shape[0, 0]) + self.cube_shape[0, 0])
        random_coords[:, 1] = (random_coords[:, 1] * (self.cube_shape[1, 1] - self.cube_shape[1, 0]) + self.cube_shape[1, 0])
        random_coords[:, 2] = (random_coords[:, 2] * (self.cube_shape[2, 1] - self.cube_shape[2, 0]) + self.cube_shape[2, 0])
        random_coords = random_coords / self.spatial_norm

        #--- pick one data
        r = np.random.choice(slices_coords.shape[0], self.boundary_batch_coords)
        slices_coords = slices_coords[r]
        slices_values = slices_values[r]
        
        samples = {'branch_input': branch_input,
                   'random_coords': random_coords,
                   'slices_coords': slices_coords,
                   'slices_values': slices_values,
                   'b_norm': b_norm}

        return samples
    

class DeepONetDatasetCNN_spatialnormalize(Dataset):

    def __init__(self, file_list, b_norm, cube_shape, boundary_batch_coords=1, random_batch_coords=1):
        super().__init__()
        self.coords_shape = cube_shape
        self.cube_shape = np.array([[0, cube_shape[0] - 1], [0, cube_shape[1] - 1], [0, cube_shape[2] - 1]])
        self.files = file_list
        self.b_norm = b_norm
        self.boundary_batch_coords = int(boundary_batch_coords)
        self.random_batch_coords = int(random_batch_coords)
        self.float_tensor = torch.FloatTensor

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        input_file = self.files[idx]
        inputs = np.load(input_file)

        branch_input = inputs['input'][..., 0].astype(np.float32)
        branch_input = branch_input / self.b_norm

        b_slices = inputs['input'].transpose(1, 2, 3, 0)
        slices_values = b_slices.reshape(-1, 3).astype(np.float32)
        slices_values = slices_values / self.b_norm      

        nx, ny, nz = self.coords_shape
        slices_coords = np.stack(np.mgrid[:b_slices.shape[0], :b_slices.shape[1], :b_slices.shape[2]], -1).astype(np.float32)
        slices_coords = slices_coords.reshape(-1, 3).astype(np.float32)
        slices_coords[..., 0] = slices_coords[..., 0] / (nx - 1)
        slices_coords[..., 1] = slices_coords[..., 1] / (ny - 1)
        slices_coords[..., 2] = slices_coords[..., 2] / (nz - 1)

        random_coords = self.float_tensor(self.random_batch_coords, 3).uniform_()

        #--- pick one data
        r = np.random.choice(slices_coords.shape[0], self.boundary_batch_coords)
        slices_coords = slices_coords[r]
        slices_values = slices_values[r]
        
        samples = {'branch_input': branch_input,
                   'random_coords': random_coords,
                   'slices_coords': slices_coords,
                   'slices_values': slices_values}

        return samples
    

class DeepONetDatasetCNN_normalize(Dataset):

    def __init__(self, file_list, cube_shape, boundary_batch_coords=1, random_batch_coords=1):
        super().__init__()
        self.coords_shape = cube_shape
        self.cube_shape = np.array([[0, cube_shape[0] - 1], [0, cube_shape[1] - 1], [0, cube_shape[2] - 1]])
        self.files = file_list
        self.boundary_batch_coords = int(boundary_batch_coords)
        self.random_batch_coords = int(random_batch_coords)
        self.float_tensor = torch.FloatTensor

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        input_file = self.files[idx]
        inputs = np.load(input_file)

        branch_input = inputs['input'][..., 0].astype(np.float32)
        b_norm = np.max(np.abs(branch_input))
        branch_input = branch_input / b_norm

        b_slices = inputs['input'].transpose(1, 2, 3, 0)
        slices_values = b_slices.reshape(-1, 3).astype(np.float32)
        slices_values = slices_values / b_norm      

        nx, ny, nz = self.coords_shape
        slices_coords = np.stack(np.mgrid[:b_slices.shape[0], :b_slices.shape[1], :b_slices.shape[2]], -1).astype(np.float32)
        slices_coords = slices_coords.reshape(-1, 3).astype(np.float32)
        slices_coords[..., 0] = slices_coords[..., 0] / (nx - 1)
        slices_coords[..., 1] = slices_coords[..., 1] / (ny - 1)
        slices_coords[..., 2] = slices_coords[..., 2] / (nz - 1)

        random_coords = self.float_tensor(self.random_batch_coords, 3).uniform_()

        #--- pick one data
        r = np.random.choice(slices_coords.shape[0], self.boundary_batch_coords)
        slices_coords = slices_coords[r]
        slices_values = slices_values[r]
        
        samples = {'branch_input': branch_input,
                   'random_coords': random_coords,
                   'slices_coords': slices_coords,
                   'slices_values': slices_values,
                   'b_norm': b_norm}

        return samples
    


class DeepONetDatasetCNNlabel(Dataset):

    def __init__(self, file_list, b_norm, spatial_norm, cube_shape, boundary_batch_coords=1, random_batch_coords=1):
        super().__init__()
        self.cube_shape = np.array([[0, cube_shape[0] - 1], [0, cube_shape[1] - 1], [0, cube_shape[2] - 1]])
        self.files = file_list
        self.b_norm = b_norm
        self.spatial_norm = spatial_norm
        self.boundary_batch_coords = int(boundary_batch_coords)
        self.random_batch_coords = int(random_batch_coords)
        self.float_tensor = torch.FloatTensor

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        input_file = self.files[idx]
        input_file = Path(input_file)
        label_file = input_file.parent.parent / 'label' / input_file.name.replace('input_', 'label_')

        inputs = np.load(input_file)
        labels = np.load(label_file)

        branch_input = inputs['input'][..., 0].astype(np.float32)
        branch_input = branch_input / self.b_norm

        slices_values = labels['label'].transpose(1, 2, 3, 0)
        b_shape = slices_values.shape
        slices_values = slices_values.reshape(-1, 3).astype(np.float32)
        slices_values = slices_values / self.b_norm        

        slices_coords = np.stack(np.mgrid[:b_shape[0], :b_shape[1], :b_shape[2]], -1).astype(np.float32)
        slices_coords = slices_coords.reshape(-1, 3).astype(np.float32)
        slices_coords = slices_coords / self.spatial_norm

        random_coords = self.float_tensor(self.random_batch_coords, 3).uniform_()
        random_coords[:, 0] = (random_coords[:, 0] * (self.cube_shape[0, 1] - self.cube_shape[0, 0]) + self.cube_shape[0, 0])
        random_coords[:, 1] = (random_coords[:, 1] * (self.cube_shape[1, 1] - self.cube_shape[1, 0]) + self.cube_shape[1, 0])
        random_coords[:, 2] = (random_coords[:, 2] * (self.cube_shape[2, 1] - self.cube_shape[2, 0]) + self.cube_shape[2, 0])
        random_coords = random_coords / self.spatial_norm

        #--- pick one data
        r = np.random.choice(slices_coords.shape[0], self.boundary_batch_coords)
        slices_coords = slices_coords[r]
        slices_values = slices_values[r]
        
        samples = {'branch_input': branch_input,
                   'random_coords': random_coords,
                   'slices_coords': slices_coords,
                   'slices_values': slices_values}

        return samples
    


class DeepONetDatasetCNNrealcoords(Dataset):

    def __init__(self, file_list, b_norm, spatial_norm, boundary_batch_coords=1, random_batch_coords=1):
        super().__init__()
        self.files = file_list
        self.b_norm = b_norm
        self.spatial_norm = spatial_norm
        self.boundary_batch_coords = int(boundary_batch_coords)
        self.random_batch_coords = int(random_batch_coords)
        self.float_tensor = torch.FloatTensor

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        input_file = Path(self.files[idx])
        label_file = input_file.parent.parent / 'label' / input_file.name.replace('input', 'label')
        inputs = np.load(input_file, mmap_mode='r')['input']
        x = np.load(label_file, mmap_mode='r')['x'] / self.spatial_norm
        y = np.load(label_file, mmap_mode='r')['y'] / self.spatial_norm
        z = np.load(label_file, mmap_mode='r')['z'] / self.spatial_norm

        branch_input = inputs[..., 0].astype(np.float32)
        branch_input = branch_input / self.b_norm

        b_slices = inputs.transpose(1, 2, 3, 0).astype(np.float32)
        slices_values = b_slices.reshape(-1, 3)
        slices_values = slices_values / self.b_norm        

        slices_coords = np.stack(np.meshgrid(x, y, [0], indexing='ij'), axis=-1).astype(np.float32)
        slices_coords = slices_coords.reshape(-1, 3)
        slices_coords = slices_coords 

        random_coords = self.float_tensor(self.random_batch_coords, 3).uniform_()
        random_coords[:, 0] = random_coords[:, 0] * (np.max(slices_coords[..., 0]) - np.min(slices_coords[..., 0])) + np.min(slices_coords[..., 0])
        random_coords[:, 1] = random_coords[:, 1] * (np.max(slices_coords[..., 1]) - np.min(slices_coords[..., 1])) + np.min(slices_coords[..., 1])
        random_coords[:, 2] = random_coords[:, 2] * (np.max(z) - np.min(z)) + np.min(z)

        #--- pick one data
        r = np.random.choice(slices_coords.shape[0], self.boundary_batch_coords)
        slices_coords = slices_coords[r]
        slices_values = slices_values[r]
        
        samples = {'branch_input': branch_input,
                   'random_coords': random_coords,
                   'slices_coords': slices_coords,
                   'slices_values': slices_values,
                   'x': x,
                   'y': y,
                   'z': z}

        return samples
    

class DeepONetDatasetCNNminmax(Dataset):

    def __init__(self, file_list, height, boundary_batch_coords=1, random_batch_coords=1):
        super().__init__()
        self.files = file_list
        self.height = height
        self.boundary_batch_coords = int(boundary_batch_coords)
        self.random_batch_coords = int(random_batch_coords)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        """
        inputs: (3, 513, 257, 1)
        """
        input_file = Path(self.files[idx])
        inputs = np.load(input_file, mmap_mode='r')['input']

        branch_input = inputs[..., 0].astype(np.float32)
        _, nx, ny = branch_input.shape
        b_min, b_max = branch_input.min(), branch_input.max()
        new_min, new_max = -1, 1
        branch_input = (branch_input - b_min) / (b_max - b_min) * (new_max - new_min) + new_min

        slices_values = branch_input.transpose(1, 2, 0)
        slices_values = slices_values.reshape(-1, 3)

        slices_coords = np.stack(np.meshgrid(
            np.linspace(0, 1, nx).reshape(-1, 1), 
            np.linspace(0, 1, ny).reshape(-1, 1), 
            np.array([0]), 
            indexing='ij'), -1).astype(np.float32)
        slices_coords = slices_coords.reshape(-1, 3)

        random_coords = torch.rand((self.random_batch_coords, 3))

        #--- pick one data
        r = np.random.choice(slices_coords.shape[0], self.boundary_batch_coords)
        slices_coords = slices_coords[r]
        slices_values = slices_values[r]
        
        samples = {'branch_input': branch_input,
                   'random_coords': random_coords,
                   'slices_coords': slices_coords,
                   'slices_values': slices_values,
                   'b_min': b_min,
                   'b_max': b_max,}

        return samples