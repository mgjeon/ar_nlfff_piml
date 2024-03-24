import torch
from torch import nn


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


# class PositionalEncoding(nn.Module):
#     def __init__(self, freqs, in_dim, out_dim):
#         super().__init__()
#         self.freqs = freqs
#         self.in_dim = in_dim
#         self.out_dim = out_dim
    
#     def forward(self, x):
#         FF = self.freqs * torch.randn(self.in_dim, self.out_dim//2, device=x.device)
#         x = torch.concatenate([torch.cos(x @ FF),
#                                torch.sin(x @ FF)], -1)
#         return x

class PositionalEncoding(nn.Module):
    """
    Positional Encoding of the input coordinates.

    encodes x to (..., sin(2^k x), cos(2^k x), ...)
    k takes "num_freqs" number of values equally spaced between [0, max_freq]
    """

    def __init__(self, max_freq, num_freqs):
        """
        Args:
            max_freq (int): maximum frequency in the positional encoding.
            num_freqs (int): number of frequencies between [0, max_freq]
        """
        super().__init__()
        freqs = 2 ** torch.linspace(0, max_freq, num_freqs)
        self.register_buffer("freqs", freqs)  # (num_freqs)

    def forward(self, x):
        """
        Inputs:
            x: (batch, num_samples, in_features)
        Outputs:
            out: (batch, num_samples, 2*num_freqs*in_features)
        """
        x_proj = x.unsqueeze(dim=-2) * self.freqs.unsqueeze(dim=-1)  # (num_rays, num_samples, num_freqs, in_features)
        x_proj = x_proj.reshape(*x.shape[:-1], -1)  # (num_rays, num_samples, num_freqs*in_features)
        out = torch.cat([torch.sin(x_proj), torch.cos(x_proj)],
                        dim=-1)  # (num_rays, num_samples, 2*num_freqs*in_features)
        return out
    

class FF_MLP(nn.Module):
    def __init__(self, max_freq, num_freqs, in_dim, out_dim, hidden_dim, num_layers):
        super().__init__()
        # self.d_in = nn.Linear(in_dim, hidden_dim)
        self.enc = nn.Sequential(
            PositionalEncoding(max_freq, num_freqs),
            nn.Linear(2*num_freqs*in_dim, hidden_dim)
        )

        lin = [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)]
        self.linear_layers = nn.ModuleList(lin)
        self.d_out = nn.Linear(hidden_dim, out_dim)
        self.activation = Sine()

    def forward(self, x):
        """
        x      : [batch_size, in_dim]
        
        output : [batch_size, out_dim]
        """

        x = self.enc(x)
        for l in self.linear_layers:
            x = self.activation(l(x))
        x = self.d_out(x)
        return x


class DeepONet(nn.Module):
    def __init__(self, branch_in_dim, trunk_in_dim, out_dim, hidden_dim, num_layers):
        super().__init__()
        self.branch_in = nn.Linear(branch_in_dim, hidden_dim)
        self.trunk_in = nn.Linear(trunk_in_dim, hidden_dim)
        lin = [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)]
        self.linear_layers = nn.ModuleList(lin)
        self.d_out = nn.Linear(hidden_dim, out_dim)
        self.activation = Sine()
    
    def forward(self, bc, x):
        """
        bc     : [batch_size, 513*257*3]
        x      : [batch_size, 3]

        output : [batch_size, 3]
        """

        u = self.activation(self.branch_in(bc))
        for l in self.linear_layers:
            u = self.activation(l(u))
        
        x = self.activation(self.trunk_in(x))
        for l in self.linear_layers:
            x = self.activation(l(x))

        y = u * x
        y = self.activation(y)
        y = self.d_out(y)
        return y
    

class DeepONetMLP(nn.Module):
    def __init__(self, branch_in_dim, trunk_in_dim, out_dim, hidden_dim, num_layers):
        super().__init__()
        self.branch_layer = MLP(branch_in_dim, hidden_dim, hidden_dim, num_layers)
        self.trunk_layer = MLP(trunk_in_dim, hidden_dim, hidden_dim, num_layers)

        self.d_out = nn.Linear(hidden_dim, out_dim)
        self.activation = Sine()
    
    def forward(self, bc, x):
        """
        bc     : [batch_size, 513*257*3]
        x      : [batch_size, batch_coords, 3]

        output : [batch_size, batch_coords, 3]
        """
        branch_latent = self.branch_layer(bc)
        trunk_latent = self.trunk_layer(x)
        latent = branch_latent[:, None, :] * trunk_latent
        output = self.d_out(self.activation(latent))
        return output
    

#------------------------------------------------------------------------------------
    

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class DeepONetCNN(nn.Module):
    def __init__(self, trunk_in_dim, out_dim, hidden_dim, num_layers):
        super().__init__()
        self.branch_inc = nn.Sequential(
            Down(3, 32),
            Down(32, 64),
            Down(64, 32),
            Down(32, 3),
        )
        self.branch_layer = MLP(3*32*16, hidden_dim, hidden_dim, num_layers)
        self.trunk_layer = MLP(trunk_in_dim, hidden_dim, hidden_dim, num_layers)

        self.d_out = nn.Linear(hidden_dim, out_dim)
        self.activation = Sine()
    
    def forward(self, bc, x):
        """
        bc     : [batch_size, 3, 513, 257]
        x      : [batch_size, batch_coords, 3]

        output : [batch_size, batch_coords, 3]
        """
        branch_latent = self.branch_inc(bc)
        branch_latent = torch.flatten(branch_latent, 1)
        branch_latent = self.branch_layer(branch_latent)
        trunk_latent = self.trunk_layer(x)
        latent = branch_latent[:, None, :] * trunk_latent
        output = self.d_out(self.activation(latent))
        return output
    

class DeepONetCNNstride(nn.Module):
    def __init__(self, trunk_in_dim, out_dim, hidden_dim, num_layers):
        super().__init__()
        self.branch_inc = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.Conv2d(32, 16, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(16),
            nn.SiLU(),
            nn.Conv2d(16, 3, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(3),
            nn.SiLU(),
        )
        self.branch_layer = MLP(3*32*16, hidden_dim, hidden_dim, num_layers)
        self.trunk_layer = MLP(trunk_in_dim, hidden_dim, hidden_dim, num_layers)

        self.d_out = nn.Linear(hidden_dim, out_dim)
        self.activation = Sine()
    
    def forward(self, bc, x):
        """
        bc     : [batch_size, 3, 513, 257]
        x      : [batch_size, batch_coords, 3]

        output : [batch_size, batch_coords, 3]
        """
        branch_latent = self.branch_inc(bc)
        branch_latent = torch.flatten(branch_latent, 1)
        branch_latent = self.branch_layer(branch_latent)
        trunk_latent = self.trunk_layer(x)
        latent = branch_latent[:, None, :] * trunk_latent
        output = self.d_out(self.activation(latent))
        return output
    

class DeepONetCNNanother(nn.Module):
    def __init__(self, trunk_in_dim, out_dim, hidden_dim, num_layers):
        super().__init__()
        self.branch_inc = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2),
            nn.SiLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1, stride=2),
            nn.SiLU(),
            nn.Conv2d(32, 3, kernel_size=3, padding=1, stride=2),
            nn.SiLU(),
        )
        self.branch_layer = nn.Linear(3*64*32, hidden_dim)
        self.trunk_layer = MLP(trunk_in_dim, hidden_dim, hidden_dim, num_layers)
        self.d_out = nn.Linear(hidden_dim, out_dim)
        self.activation = Sine()
    
    def forward(self, bc, x):
        """
        bc     : [batch_size, 3, 513, 257]
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
    
class DeepONetCNNanother_FF(nn.Module):
    def __init__(self, max_freq, num_freqs, trunk_in_dim, out_dim, hidden_dim, num_layers):
        super().__init__()
        self.branch_inc = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2),
            nn.SiLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1, stride=2),
            nn.SiLU(),
            nn.Conv2d(32, 3, kernel_size=3, padding=1, stride=2),
            nn.SiLU(),
        )
        self.branch_layer = nn.Linear(3*64*32, hidden_dim)
        # self.enc = nn.Sequential(
        #     PositionalEncoding(max_freq, num_freqs),
        #     nn.Linear(2*num_freqs*trunk_in_dim, hidden_dim)
        # )
        self.trunk_layer = FF_MLP(max_freq, num_freqs, trunk_in_dim, hidden_dim, hidden_dim, num_layers)
        self.d_out = nn.Linear(hidden_dim, out_dim)
        self.activation = Sine()
    
    def forward(self, bc, x):
        """
        bc     : [batch_size, 3, 513, 257]
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
    

class DeepONetCNNanotherSine(nn.Module):
    def __init__(self, trunk_in_dim, out_dim, hidden_dim, num_layers):
        super().__init__()
        self.branch_inc = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3),
            Sine(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2),
            Sine(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1, stride=2),
            Sine(),
            nn.Conv2d(32, 3, kernel_size=3, padding=1, stride=2),
            Sine(),
        )
        self.branch_layer = nn.Linear(3*64*32, hidden_dim)
        self.trunk_layer = MLP(trunk_in_dim, hidden_dim, hidden_dim, num_layers)
        self.d_out = nn.Linear(hidden_dim, out_dim)
        self.activation = Sine()
    
    def forward(self, bc, x):
        """
        bc     : [batch_size, 3, 513, 257]
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
    
