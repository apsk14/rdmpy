"""
Implements a basic UNet model for image deblurring.

Adapted from https://github.com/SecretMG/UNet-for-Image-Denoising
"""

import torch
from torch import nn
import torch.nn.functional as F


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
        self.layer = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.layer(x)


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
        # print('XSHAPE:', x.shape, 'XSHORTCUTSHAPE:', x_shortcut.shape)
        x = torch.cat((x_shortcut, x), dim=1)
        x = self.layer3(x)
        return x


class UNet(nn.Module):
    def __init__(self, scale):
        super(UNet, self).__init__()
        self.is_seidel = False
        self.model_name = "UNET_basic"
        scaled = lambda x: int(x * scale)

        self.inc1 = DoubleConv(2, scaled(16))
        self.inc2 = DoubleConv(scaled(16), scaled(64))
        self.down0 = Down_DC(scaled(32), scaled(64))
        self.down1 = Down_DC(scaled(64), scaled(128))
        self.down2 = Down_DC(scaled(128), scaled(256))
        self.down3 = Down_DC(scaled(256), scaled(512))
        self.down4 = Down_DC(scaled(512), scaled(1024))
        self.down5 = DoubleConv(scaled(1024), scaled(1024))
        self.mp = nn.MaxPool2d(2)

        self.up4 = Up_DC(scaled(1024), scaled(512))
        self.up3 = Up_DC(scaled(512), scaled(256))
        self.up2 = Up_DC(scaled(256), scaled(128))
        self.up1 = Up_DC(scaled(128), scaled(64))
        self.up0 = Up_DC(scaled(64), scaled(32))

        self.out = nn.Conv2d(scaled(32), 1, kernel_size=1)
        self.cleanup = nn.Conv2d(1, 1, kernel_size=7, padding=3)

    def forward(self, x):
        resolution = x.shape[-1]
        x = x.view(-1, 2, resolution, resolution)

        x0 = self.inc1(x)
        x1 = self.mp(x0)
        x1 = self.inc2(x1)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.down5(x5)

        x = self.up4(x, x4)
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        x = self.up0(x, x0.repeat(1, 2, 1, 1))
        x = self.out(x)
        return x.view(-1, resolution, resolution)
