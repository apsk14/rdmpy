"""
Implementation of DeepRD.
DeepRD uses HyperNetworks to generate spatially-varying convolutional filters.
Adapted from https://github.com/SecretMG/UNet-for-Image-Denoising, https://github.com/g1910/HyperNetworks
"""
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from .hypernetwork_modules import HyperNetwork
from .InferenceUNET import InferenceUNet
import math
from ..._src import polar_transform


class DoubleConv(nn.Module):
    def __init__(
        self, in_channels, out_channels, hypernet, seidel_dim=6, hyper_dim=128
    ):
        super(DoubleConv, self).__init__()
        self.hypernet = hypernet
        self.in_channels, self.out_channels = in_channels, out_channels
        hyper_out_size = 64
        self.k, self.h = in_channels // hyper_out_size, out_channels // hyper_out_size

        self.seidel_block = self.k * self.h > 0

        if not self.seidel_block:
            self.seidel_block = False
            self.layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=0),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=0),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            )
            return

        # This HyperUNet uses sets of many small MLPs to produce embedding for each convolutional block
        self.nn_list_one = torch.nn.ModuleList([])
        self.nn_list_two = torch.nn.ModuleList([])
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        for _ in range(self.h * self.k):
            seq = nn.Sequential(
                nn.Linear(seidel_dim, hyper_dim),
                nn.ReLU(),
                nn.Dropout(p=0.1),
                nn.Linear(hyper_dim, hyper_dim),
                nn.ReLU(),
            )
            self.nn_list_one.append(seq)

        for _ in range(self.h * self.h):
            seq = nn.Sequential(
                nn.Linear(seidel_dim, hyper_dim),
                nn.ReLU(),
                nn.Dropout(p=0.1),
                nn.Linear(hyper_dim, hyper_dim),
                nn.ReLU(),
            )
            self.nn_list_two.append(seq)

    def lri_conv_functional(self, x, weights, padding=None):
        c_out, c_in, num_radii, k, _ = weights.shape
        padding = k // 2 if not padding else padding
        b, _, h, w = x.shape

        # (padding_left,padding_right, padding_top, padding_bottom)
        x = F.pad(x, (0, 0, padding, padding), mode="circular")
        x = F.pad(x, (padding, padding, 0, 0), mode="constant")

        x_unfolded = F.unfold(x, (k, k), padding=0)
        x_unfolded = x_unfolded.transpose(1, 2).view(b, 1, h, w, -1)
        interp_weights = F.interpolate(
            weights, (w, *weights.shape[-2:]), mode="trilinear"
        )
        x_unfolded = x_unfolded.flatten(0, 2)
        weights_unfolded = torch.reshape(interp_weights.transpose(1, 2), (c_out, w, -1))
        out = torch.einsum(
            "...ij,...ij->...i", (x_unfolded[:, None, ...], weights_unfolded[None, ...])
        )
        out = out.view(b, h, c_out, w).transpose(1, 2)
        return out

    def forward(self, x, seidel=None):
        if not self.seidel_block:
            x = F.pad(x, (0, 0, 2, 2), mode="circular")
            x = F.pad(x, (2, 2, 0, 0), mode="constant")
            return self.layer(x)

        out_acc = []
        for i in range(seidel.shape[0]):
            w1, w2 = self.generate_weights(seidel[i : i + 1])
            # print("WEIGHTS SHAPES:", w1.shape, w2.shape)

            # batch_x = F.conv2d(x[i:i+1], w1, padding=1)
            batch_x = self.lri_conv_functional(x[i : i + 1], w1, padding=1)

            batch_x = self.bn1(batch_x)
            batch_x = F.relu(batch_x)

            # batch_x = F.conv2d(batch_x, w2, padding=1)
            batch_x = self.lri_conv_functional(batch_x, w2, padding=1)

            batch_x = self.bn2(batch_x)
            batch_x = F.relu(batch_x)
            out_acc.append(batch_x)

        out = torch.cat(out_acc, dim=0)
        return out

    def generate_weights(self, seidel):
        ww = []
        for i in range(self.h):
            w = []
            for j in range(self.k):
                embedding = self.nn_list_one[i * self.k + j](seidel)
                w.append(self.hypernet(embedding))
            ww.append(torch.cat(w, dim=1))
        w1 = torch.cat(ww, dim=0)

        ww = []
        for i in range(self.h):
            w = []
            for j in range(self.h):
                embedding = self.nn_list_two[i * self.h + j](seidel)
                w.append(self.hypernet(embedding))
            ww.append(torch.cat(w, dim=1))
        w2 = torch.cat(ww, dim=0)

        return (w1, w2)

    def get_conv_filters(self, seidel):
        if self.seidel_block:
            return self.generate_weights(seidel)
        else:
            return self.layer.weight

    def set_weights(self, inference_block, seidel):
        new_seq = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(),
        )
        if self.seidel_block:
            w1, w2 = self.generate_weights(seidel)
            new_seq[0].weight = torch.nn.Parameter(w1, requires_grad=False)
            new_seq[1].weight = torch.nn.Parameter(self.bn1.weight, requires_grad=False)
            new_seq[3].weight = torch.nn.Parameter(w2, requires_grad=False)
            new_seq[4].weight = torch.nn.Parameter(self.bn2.weight, requires_grad=False)
        else:
            for i in [0, 1, 3, 4]:
                new_seq[i].weight = torch.nn.Parameter(
                    self.layer[i].weight, requires_grad=False
                )

        for param in new_seq.parameters():
            param.requires_grad = False

        inference_block.layer = new_seq


# Max pool to half spatial dim DoubleConv
class Down_DC(nn.Module):
    def __init__(self, in_channels, out_channels, hypernet, seidel_dim=6):
        super(Down_DC, self).__init__()
        self.pool = nn.MaxPool2d(2)
        self.layer1 = DoubleConv(
            in_channels, out_channels, hypernet, seidel_dim=seidel_dim
        )

    def forward(self, x, seidel):
        x = self.pool(x)
        x = self.layer1(x, seidel)
        return x

    def get_conv_filters(self, seidel):
        return self.layer1.generate_weights(seidel)

    def set_weights(self, inference_block, seidel):
        self.layer1.set_weights(inference_block.layer1, seidel)


# Interpolate to double spatial dim DoubleConv
class Up_DC(nn.Module):
    def __init__(self, in_channels, out_channels, hypernet, seidel_dim=6):
        super(Up_DC, self).__init__()
        self.dim_output = out_channels
        self.layer1 = F.interpolate
        self.layer2 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.layer3 = DoubleConv(
            in_channels, out_channels, hypernet, seidel_dim=seidel_dim
        )

    def forward(self, x, x_shortcut, seidel):
        x = self.layer1(x, scale_factor=2)
        x = self.layer2(x)
        x = torch.cat((x_shortcut, x), dim=1)
        x = self.layer3(x, seidel)
        return x

    def get_conv_filters(self, seidel):
        return self.layer3.generate_weights(seidel)

    def set_weights(self, inference_block, seidel):
        inference_block.layer2.weight = torch.nn.Parameter(
            self.layer2.weight, requires_grad=False
        )
        self.layer3.set_weights(inference_block.layer3, seidel)


class UNet(nn.Module):
    def __init__(self, scale=1.0, num_frequencies=8):
        super(UNet, self).__init__()

        def Log2(x):
            return math.log10(x) / math.log10(2)

        if math.ceil(Log2(scale)) != math.floor(Log2(scale)):
            raise ValueError("Scale must be a power of 2")

        self.scale = scale
        self.num_frequencies = num_frequencies
        self.seidel_dim = 6 * num_frequencies
        self.is_seidel = True
        self.model_name = "Hyper_UNET_LRI"
        self.hypernet = HyperNetwork(z_dim=128, f_size=3)
        scaled = lambda x: int(x * scale)

        self.inc = DoubleConv(2, scaled(32), self.hypernet, seidel_dim=self.seidel_dim)
        self.inc1 = DoubleConv(2, scaled(16), self.hypernet, seidel_dim=self.seidel_dim)
        self.inc2 = DoubleConv(
            scaled(16), scaled(32), self.hypernet, seidel_dim=self.seidel_dim
        )
        self.down0 = Down_DC(
            scaled(32), scaled(64), self.hypernet, seidel_dim=self.seidel_dim
        )
        self.down1 = Down_DC(
            scaled(32), scaled(64), self.hypernet, seidel_dim=self.seidel_dim
        )
        self.down2 = Down_DC(
            scaled(64), scaled(128), self.hypernet, seidel_dim=self.seidel_dim
        )
        self.down3 = Down_DC(
            scaled(128), scaled(256), self.hypernet, seidel_dim=self.seidel_dim
        )
        self.down4 = Down_DC(
            scaled(256), scaled(512), self.hypernet, seidel_dim=self.seidel_dim
        )
        self.down5 = DoubleConv(
            scaled(512), scaled(512), self.hypernet, seidel_dim=self.seidel_dim
        )
        self.mp = nn.MaxPool2d(2)

        self.up4 = Up_DC(
            scaled(512), scaled(256), self.hypernet, seidel_dim=self.seidel_dim
        )
        self.up3 = Up_DC(
            scaled(256), scaled(128), self.hypernet, seidel_dim=self.seidel_dim
        )
        self.up2 = Up_DC(
            scaled(128), scaled(64), self.hypernet, seidel_dim=self.seidel_dim
        )
        self.up1 = Up_DC(
            scaled(64), scaled(32), self.hypernet, seidel_dim=self.seidel_dim
        )
        self.up0 = Up_DC(
            scaled(32), scaled(16), self.hypernet, seidel_dim=self.seidel_dim
        )
        self.out = nn.Conv2d(scaled(16), 1, kernel_size=1)
        # OLD
        # self.cleanup = nn.Conv2d(1, 1, kernel_size=7, padding=3)

        self.cleanup1 = nn.Conv2d(1, 8, kernel_size=7, padding=3)
        self.cleanup_bn1 = nn.BatchNorm2d(8)
        self.cleanup2 = nn.Conv2d(8, 8, kernel_size=7, padding=3)
        self.cleanup_bn2 = nn.BatchNorm2d(8)
        self.cleanup3 = nn.Conv2d(8, 1, kernel_size=7, padding=3)

    def forward(self, x, seidel):
        resolution = x.shape[-1]
        x = x.view(-1, 2, resolution, resolution)

        # From https://github.com/nerfstudio-project/nerfstudio
        freqs = 2 ** torch.linspace(
            0.0, self.num_frequencies, self.num_frequencies, device=x.device
        )
        seidel_encoded = seidel[..., None] * freqs
        seidel_encoded = seidel_encoded.view(*seidel_encoded.shape[:-2], -1)

        # Cartesian -> polar transformation
        batch_dim = x.shape[0]
        if resolution > 700:
            x_polar = torch.zeros(batch_dim, 2, resolution * 2, resolution).to(x)
        else:
            x_polar = torch.zeros(batch_dim, 2, resolution * 4, resolution).to(x)
        for batch_idx in range(batch_dim):
            batch_blur, batch_lsi = x[batch_idx]
            batch_polar = torch.stack(
                (
                    polar_transform.img2polar(batch_blur, resolution),
                    polar_transform.img2polar(batch_lsi, resolution),
                )
            )
            # print("BATCH POLAR SHAPE:", batch_polar.shape, "X SHAPE:", x_polar.shape)
            x_polar[batch_idx] = batch_polar
        x = x_polar
        del x_polar

        x0 = self.inc1(x)
        x1 = self.mp(x0)
        x1 = self.inc2(x1)
        x2 = self.down1(x1, seidel_encoded)
        x3 = self.down2(x2, seidel_encoded)
        x4 = self.down3(x3, seidel_encoded)
        x5 = self.down4(x4, seidel_encoded)
        x = self.down5(x5, seidel_encoded)
        x = self.up4(x, x4, seidel_encoded)
        x = self.up3(x, x3, seidel_encoded)
        x = self.up2(x, x2, seidel_encoded)
        x = self.up1(x, x1, seidel_encoded)
        x = self.up0(x, x0, seidel_encoded)
        x = self.out(x)
        if resolution > 700:
            x = x.view(-1, resolution * 2, resolution)
        else:
            x = x.view(-1, resolution * 4, resolution)

        # Polar -> cartesian transformation
        x += 0.5
        x_out = torch.zeros(batch_dim, resolution, resolution).to(x)
        for batch_idx in range(batch_dim):
            x_out[batch_idx] = polar_transform.polar2img(
                x[batch_idx], [resolution, resolution]
            )
        x_out -= 0.5

        # x_out = self.cleanup(x_out[:, None, ...])
        x_out = self.cleanup1(x_out[:, None, ...])
        x_out = F.relu(self.cleanup_bn1(x_out))
        x_out = self.cleanup2(x_out)
        x_out = F.relu(self.cleanup_bn2(x_out))
        x_out = self.cleanup3(x_out)

        return x_out.view(-1, resolution, resolution)

    def load_my_state_dict(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                continue
            if isinstance(param, Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            own_state[name].copy_(param)

    def get_conv_filters(self, seidel, i):
        non_seidel_blocks = {0, 1, 9, 10}
        if i in non_seidel_blocks:
            raise ValueError(
                f"Convolutional layer at index {i} is not generated from hypernet."
            )
        index_to_layer = {
            2: self.down2,
            3: self.down3,
            4: self.down4,
            5: self.down5,
            6: self.up4,
            7: self.up3,
            8: self.up2,
        }

        return index_to_layer[i].get_conv_filters(seidel)

    # Doesn't work at the moment
    def get_inference_unet(self, seidel, device):
        new_model = InferenceUNet(self.scale)
        self.inc.set_weights(new_model.inc, seidel)
        self.down1.set_weights(new_model.down1, seidel)
        self.down2.set_weights(new_model.down2, seidel)
        self.down3.set_weights(new_model.down3, seidel)
        self.down4.set_weights(new_model.down4, seidel)
        self.down5.set_weights(new_model.down5, seidel)
        self.up4.set_weights(new_model.up4, seidel)
        self.up3.set_weights(new_model.up3, seidel)
        self.up2.set_weights(new_model.up2, seidel)
        self.up1.set_weights(new_model.up1, seidel)
        new_model.out.weight = torch.nn.Parameter(self.out.weight, requires_grad=False)
        for param in new_model.parameters():
            param.requires_grad = False
        new_model.to(device)
        return new_model

    # Doesn't work at the moment
    def get_interpolation_unet(self, i, seidel_one, seidel_two, device):
        model_one = self.get_inference_unet(seidel_one, device)
        model_two = self.get_inference_unet(seidel_two, device)
        for param_one, param_two in zip(model_one.parameters(), model_two.parameters()):
            param_one = (i) * param_one + (i - 1) * param_two
        return model_one
