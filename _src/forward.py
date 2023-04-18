# code containing methods utilizing the linear revolution invariant:
# import polarTransform as pt
import numpy as np
from torch_fftconv import fft_conv2d
from . import util, seidel, polar_transform
from torch.nn.functional import pad
from scipy.signal import fftconvolve, convolve2d

# from . import pt_old as polar_transform
from tqdm import tqdm
import torch
import torch.fft as fft
import pdb
from scipy.ndimage import shift
import matplotlib.pyplot as plt


def lri(
    obj,
    psf_tensor_fft,
    method="normal",
    device=torch.device("cpu"),
    verbose=False,
    diff=None,
    calib=False,
):

    # infer info from the PSF roft
    num_radii = psf_tensor_fft.shape[0]

    # get object RoFT
    obj_polar = polar_transform.img2polar(obj, numRadii=num_radii)

    r_list = np.sqrt(2) * (
        np.linspace(0, (obj.shape[0] / 2), num_radii, endpoint=False, retstep=False)
        + 0.5
    )

    dr = r_list[1] - r_list[0]
    dtheta = 2 * np.pi / obj_polar.shape[0]

    obj_fft = fft.rfft(obj_polar, dim=0)

    # create blank image RoFT which will be subsequently filled in
    img_polar_fft = torch.zeros_like(obj_fft, dtype=torch.complex64, device=device)
    if method == "alt":
        r_list = torch.tensor(r_list, device=device).float()
        print("Using the alt (matrix based) method")
        freqs = (
            tqdm(range(psf_tensor_fft.shape[1]))
            if verbose
            else (range(psf_tensor_fft.shape[1]))
        )
        for i in freqs:
            psf_slice = psf_tensor_fft[:, i, :].T
            recon_row = torch.matmul(psf_slice, (obj_fft[i, :] * r_list)[:, None])
            img_polar_fft[i, :] = recon_row.T
    else:
        radii = tqdm(enumerate(r_list)) if verbose else (enumerate(r_list))
        for index, r in radii:
            curr_psf_polar_fft = psf_tensor_fft[index, :, :]
            # curr_psf_polar_fft = curr_psf_polar_fft/curr_psf_polar_fft.abs().sum()
            integration_area = r * dr * dtheta
            img_polar_fft += integration_area * (
                obj_fft[:, index][:, None] * curr_psf_polar_fft
            )

    img_polar = fft.irfft(img_polar_fft, dim=0)
    img = polar_transform.polar2img(img_polar, obj.shape)

    return img, obj_polar


def lsi(obj, psf):
    img = fft.irfftn(fft.rfftn(obj) * fft.rfftn(psf))
    return torch.roll(img, shifts=(-img.shape[0] // 2, -img.shape[1] // 2), dims=(0, 1))


def padded_lsi(obj, psf):

    extent = obj.shape[0]
    padded_obj = pad(obj, (0, extent, 0, extent))
    padded_psf = pad(psf, (0, extent, 0, extent))
    padded_img = fft.irfftn(fft.rfftn(padded_obj) * fft.rfftn(padded_psf))
    padded_img = torch.roll(
        padded_img,
        shifts=(-padded_img.shape[0] // 4, -padded_img.shape[1] // 4),
        dims=(0, 1),
    )

    return padded_img[0:-extent, 0:-extent]

