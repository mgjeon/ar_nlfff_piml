import torch
from torch import nn
from neuralop.models import UNO

class Sine(nn.Module):
    def __init__(self, w0=1.):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, num_layers):
        super().__init__()
        self.d_in = nn.Linear(in_dim, hidden_dim)
        lin = [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)]
        self.linear_layers = nn.ModuleList(lin)
        self.d_out = nn.Linear(hidden_dim, out_dim)
        self.activation = Sine()

    def forward(self, x):
        """
        x      : [batch_size, in_dim]
        
        output : [batch_size, out_dim]
        """

        x = self.activation(self.d_in(x))
        for l in self.linear_layers:
            x = self.activation(l(x))
        x = self.d_out(x)
        return x
    

class DeepONet(nn.Module):
    def __init__(self, trunk_in_dim, out_dim, latent_dim, hidden_dim, num_layers):
        super().__init__()
        self.branch_inc = UNO(
            in_channels=3,
            out_channels=64,
            hidden_channels=64,
            lifting_channels=64,
            projection_channels=64,
            n_layers = 4,
            uno_n_modes = [[8,8],
                           [8,8],
                           [8,8],
                           [8,8]],
            uno_out_channels = [64,
                                64,
                                64,
                                64],
            uno_scalings =  [[0.5,0.5],
                            [0.5,0.5],
                            [0.5,0.5],
                            [0.5,0.5],]
        )
        self.branch_layer = nn.Linear(64*32*16, latent_dim)
        self.trunk_layer = MLP(trunk_in_dim, latent_dim, hidden_dim, num_layers)
        self.d_out = nn.Linear(latent_dim, out_dim)
        self.activation = Sine()
    
    def forward(self, bc, x):
        """
        bc     : [batch_size, 3, 512, 256]
        x      : [batch_size, batch_coords, trunk_in_dim]

        output : [batch_size, batch_coords, out_dim]
        """
        branch_latent = self.branch_inc(bc)
        branch_latent = torch.flatten(branch_latent, 1)
        branch_latent = self.branch_layer(branch_latent)
        trunk_latent = self.trunk_layer(x)
        latent = branch_latent[:, None, :] * trunk_latent
        output = self.d_out(self.activation(latent))
        return output