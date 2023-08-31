"""
Hypernetwork modules for DeepRD, based on "HyperNetworks" by Ha et al. (2017)
"""

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


# Hypernetwork to generate convolutional kernels for all conditioned U-Net blocks
class HyperNetwork(nn.Module):
    def __init__(self, f_size=3, z_dim=128, out_size=64, in_size=64, num_radii=2):
        super(HyperNetwork, self).__init__()
        self.z_dim = z_dim
        self.f_size = f_size
        self.out_size = out_size
        self.in_size = in_size
        self.num_radii = num_radii

        # Tensors of random 0/1
        self.w1 = Parameter(
            torch.fmod(
                torch.randn(
                    (
                        self.num_radii,
                        self.z_dim,
                        self.out_size * self.f_size * self.f_size,
                    )
                ),  # .cuda(),
                2,
            )
        )
        self.b1 = Parameter(
            torch.fmod(torch.randn((1, self.out_size * self.f_size * self.f_size)), 2)
        )

        self.w2 = Parameter(
            torch.fmod(torch.randn((self.z_dim, self.in_size * self.z_dim)), 2)
        )
        self.b2 = Parameter(torch.fmod(torch.randn((self.in_size * self.z_dim)), 2))

    def forward(self, z):
        # print("z shape:", z.shape)

        # print("matmul1 shape:", torch.matmul(z, self.w2).shape)

        h_in = torch.matmul(z, self.w2) + self.b2
        h_in = h_in.view(self.in_size, self.z_dim)

        # print("h_in shape:", h_in.shape)

        h_final = torch.matmul(h_in, self.w1) + self.b1
        # print("h_final shape:", h_final.shape)
        kernel = h_final.view(
            self.num_radii, self.out_size, self.in_size, self.f_size, self.f_size
        ).permute(1, 2, 0, 3, 4)
        # print("kernel shape:", kernel.shape)
        # print()

        return kernel
