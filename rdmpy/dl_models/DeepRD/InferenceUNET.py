"""
Deblurring network output by the Hypernetwork
"""

import torch
from torch import nn
import torch.nn.functional as F
from ..._src import polar_transform

"--- InferenceUNet"


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.layer(x)


class Down_DC(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down_DC, self).__init__()
        self.pool = nn.MaxPool2d(2)
        self.layer1 = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        x = self.pool(x)
        x = self.layer1(x)
        return x


class Up_DC(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up_DC, self).__init__()
        self.dim_output = out_channels
        self.layer1 = F.interpolate
        self.layer2 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.layer3 = DoubleConv(in_channels, out_channels)

    def forward(self, x, x_shortcut):
        x = self.layer1(x, scale_factor=2)
        x = self.layer2(x)
        x = torch.cat((x_shortcut, x), dim=1)
        x = self.layer3(x)
        return x


# InferenceUNet functionality is untested with latest HyperUNet changes!!!
class InferenceUNet(nn.Module):
    def __init__(self, scale):
        super(InferenceUNet, self).__init__()
        self.is_seidel = False
        self.model_name = "InferenceUNet"
        self.inc = DoubleConv(2, int(32 * scale))
        self.down1 = Down_DC(int(32 * scale), int(64 * scale))
        self.down2 = Down_DC(int(64 * scale), int(128 * scale))
        self.down3 = Down_DC(int(128 * scale), int(256 * scale))
        self.down4 = Down_DC(int(256 * scale), int(512 * scale))
        self.down5 = DoubleConv(int(512 * scale), int(512 * scale))
        self.up4 = Up_DC(int(512 * scale), int(256 * scale))
        self.up3 = Up_DC(int(256 * scale), int(128 * scale))
        self.up2 = Up_DC(int(128 * scale), int(64 * scale))
        self.up1 = Up_DC(int(64 * scale), int(32 * scale))
        self.out = nn.Conv2d(int(32 * scale), 1, kernel_size=1)

    def forward(self, x):
        x = x.view(-1, 2, 512, 512)

        batch_dim = x.shape[0]
        x_polar = torch.zeros(batch_dim, 2, 2048, 512).to(x)
        for batch_idx in range(batch_dim):
            batch_blur, batch_lsi = x[batch_idx]
            batch_polar = torch.stack(
                (
                    polar_transform.img2polar(batch_blur, 512),
                    polar_transform.img2polar(batch_lsi, 512),
                )
            )
            x_polar[batch_idx] = batch_polar
        x = x_polar
        del x_polar
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x5 = self.down4(x4)
        x = self.down5(x5)

        x = self.up4(x, x4)
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        x = self.out(x)
        x = x.view(-1, 2048, 512)

        x += 0.5
        x_out = torch.zeros(batch_dim, 512, 512).to(x)
        for batch_idx in range(batch_dim):
            x_out[batch_idx] = polar_transform.polar2img(x[batch_idx], [512, 512])
        x_out -= 0.5

        return x_out.view(-1, 512, 512)
