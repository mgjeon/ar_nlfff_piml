import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path 

class ISEEDataset_Hnorm_Unit_Aug_Potential(Dataset):

    def __init__(self, data_path, b_norm):
        files = list(Path(data_path).glob('**/input/*.npz'))
        self.files = sorted([f for f in files])
        self.b_norm = b_norm
        self.length = len(self.files)

    def __len__(self):
        return 2*self.length

    def __getitem__(self, idx):
        is_flip = False
        if idx >= self.length:
            is_flip = True
            idx = idx - self.length

        # Assume unit dx, dy, dz
        dx = torch.from_numpy(np.array([1.0]).astype(np.float32)).reshape(-1, 1)
        dy = torch.from_numpy(np.array([1.0]).astype(np.float32)).reshape(-1, 1)
        dz = torch.from_numpy(np.array([1.0]).astype(np.float32)).reshape(-1, 1)

        label_file = self.files[idx].parent.parent / 'label' / f'label_{self.files[idx].stem[6:]}.npz'
        #           [3,  NX,  NY,  NZ]
        # NLFFF     [3, 513, 257, 257]
        labels = torch.from_numpy(np.load(label_file, mmap_mode='r')['pot'].astype(np.float32))
        divisor = (self.b_norm / np.arange(1, labels.shape[-1] + 1)).reshape(1, 1, 1, -1).astype(np.float32)
        labels = labels / divisor
        # NLFFF     [3, 512, 256, 256]  remove duplicated periodic boundary
        labels = labels[:, :-1, :-1, :-1]

        #           [3,  NX,  NY,  1]
        inputs = labels[:, :, :, 0][:, :, :, None]

        if is_flip:
            # rotation by 180 deg around z-axis
            inputs = torch.flip(inputs, dims=(1, 2))
            inputs[0] = -inputs[0]
            inputs[1] = -inputs[1]
            labels = torch.flip(labels, dims=(1, 2))
            labels[0] = -labels[0]
            labels[1] = -labels[1]

        samples = {'input': inputs, 'label': labels,
                   'label_name': label_file.stem,
                   'dx': dx, 'dy': dy, 'dz': dz}

        return samples
    
    
class ISEEDataset_Multiple_Hnorm_Unit_Aug_Potential(Dataset):

    def __init__(self, dataset_path, b_norm, test_noaa=None):
        self.files = list(Path(dataset_path).glob('**/input/*.npz'))
        if isinstance(test_noaa, str):
            self.files = sorted([f for f in self.files if not test_noaa == str(f.parent.parent.stem)])
        elif isinstance(test_noaa, list):
            self.files = sorted([f for f in self.files if not test_noaa[0] == str(f.parent.parent.stem)])
            for t_noaa in test_noaa[1:]:
                self.files = sorted([f for f in self.files if not t_noaa == str(f.parent.parent.stem)])
        self.b_norm = b_norm
        self.length = len(self.files)

    def __len__(self):
        return 2*self.length

    def __getitem__(self, idx):
        is_flip = False
        if idx >= self.length:
            is_flip = True
            idx = idx - self.length
        
        dx = torch.from_numpy(np.array([1.0]).astype(np.float32)).reshape(-1, 1)
        dy = torch.from_numpy(np.array([1.0]).astype(np.float32)).reshape(-1, 1)
        dz = torch.from_numpy(np.array([1.0]).astype(np.float32)).reshape(-1, 1)

        label_file = self.files[idx].parent.parent / 'label' / f'label_{self.files[idx].stem[6:]}.npz'
        # NLFFF     [3, 513, 257, 257]
        labels = torch.from_numpy(np.load(label_file, mmap_mode='r')['pot'].astype(np.float32))
        divisor = (self.b_norm / np.arange(1, labels.shape[-1] + 1)).reshape(1, 1, 1, -1).astype(np.float32)
        labels = labels / divisor
        # NLFFF     [3, 512, 256, 256]  remove duplicated periodic boundary
        labels = labels[:, :-1, :-1, :-1]

        inputs = labels[:, :, :, 0][:, :, :, None]

        if is_flip:
            # rotation by 180 deg around z-axis
            inputs = torch.flip(inputs, dims=(1, 2))
            inputs[0] = -inputs[0]
            inputs[1] = -inputs[1]
            labels = torch.flip(labels, dims=(1, 2))
            labels[0] = -labels[0]
            labels[1] = -labels[1]

        samples = {'input': inputs, 'label': labels, 
                   'label_name': label_file.stem,
                   'dx': dx, 'dy': dy, 'dz': dz}

        return samples