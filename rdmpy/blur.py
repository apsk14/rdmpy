"""Implementations of blur models"""

import numpy as np
import torch
import torch.fft as fft
from torch.nn.functional import pad

from tqdm import tqdm

from ._src import polar_transform, seidel, util


def ring_convolve(
    obj,
    psf_roft,
    device=torch.device("cpu"),
    verbose=False,
):
    """

    Returns the ring convolution of an object with a stack of PSFs.

    Parameters
    ----------
    obj : torch.Tensor
        The image to be convolved with the PSF stack. Must be (N,N).

    psf_roft : torch.Tensor
        The stack of PSFs to convolve the image with. The PSFs should be in the
        Rotational Fourier domain. Should be (N, M, L) where N is the number of PSFs,
        M is the number of angles, and is L is number of radii the in the RoFT.

    device : torch.device, optional
        The device to use for the computation.

    verbose : bool, optional
        Whether to display a progress bar.

    Returns
    -------
    img : torch.Tensor
        The ring convolution of the object with the PSF stack. Will be (N,N).

    """

    if not torch.is_tensor(obj):
        obj = torch.tensor(obj).float()
    if obj.device is not device:
        obj = obj.to(device)
    if not len(obj.shape) == 2 and obj.shape[0] == obj.shape[1]:
        raise AssertionError(f"Object of shape {obj.shape} must be 2d square image")

    # infer info from the PSF roft
    num_radii = psf_roft.shape[0]

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

    radii = tqdm(enumerate(r_list)) if verbose else (enumerate(r_list))
    for index, r in radii:
        curr_psf_polar_fft = (
            psf_roft[index, 0 : psf_roft.shape[1] // 2, :]
            + 1j * psf_roft[index, psf_roft.shape[1] // 2 :, :]
        )
        # curr_psf_polar_fft = psf_roft[index, :, :]
        integration_area = r * dr * dtheta
        img_polar_fft += integration_area * (
            obj_fft[:, index][:, None] * curr_psf_polar_fft
        )
        del curr_psf_polar_fft

    img_polar = fft.irfft(img_polar_fft, dim=0)
    img = polar_transform.polar2img(img_polar, obj.shape)

    return img


def ring_convolve_fractional(
    obj, psf_roft, fraction, device=torch.device("cpu"), verbose=False
):
    """

    Returns the ring convolution of an object with a stack of PSFs.

    Parameters
    ----------
    obj : torch.Tensor
        The image to be convolved with the PSF stack. Must be (N,N).

    psf_roft : torch.Tensor
        The stack of PSFs to convolve the image with. The PSFs should be in the
        Rotational Fourier domain. Should be (N, M, L) where N is the number of PSFs,
        M is the number of angles, and is L is number of radii the in the RoFT.

    fraction : list of two integers
        The fraction of the PSF stack to use. The first integer is which component and
        the second integer is how many components to use.

    device : torch.device, optional
        The device to use for the computation.

    verbose : bool, optional
        Whether to display a progress bar.

    Returns
    -------
    img : torch.Tensor
        The ring convolution of the object with the PSF stack. Will be (N,N).

    """

    if not torch.is_tensor(obj):
        obj = torch.tensor(obj).float()
    if obj.device is not device:
        obj = obj.to(device)

    # infer info from the PSF roft
    num_radii = psf_roft.shape[0]

    # get object RoFT
    obj_polar = polar_transform.img2polar(obj, numRadii=num_radii)

    r_list = np.sqrt(2) * (
        np.linspace(0, (obj.shape[0] / 2), num_radii, endpoint=False, retstep=False)
        + 0.5
    )

    obj_fft = fft.rfft(obj_polar, dim=0)

    # create blank image RoFT which will be subsequently filled in
    img_polar_fft = torch.zeros_like(obj_fft, dtype=torch.complex64, device=device)

    r_list, rs = util.split_list(r_list, fraction[1], fraction[0])
    dr = r_list[1] - r_list[0]
    dtheta = 2 * np.pi / obj_polar.shape[0]

    radii = tqdm(r_list) if verbose else (r_list)
    for index, r in enumerate(radii):
        curr_psf_polar_fft = (
            psf_roft[rs, 0 : psf_roft.shape[1] // 2, :]
            + 1j * psf_roft[rs, psf_roft.shape[1] // 2 :, :]
        )  #
        # curr_psf_polar_fft = psf_roft[index, :, :]
        integration_area = r * dr * dtheta
        img_polar_fft += integration_area * (
            obj_fft[:, rs][:, None] * curr_psf_polar_fft
        )
        rs = rs + 1
        del curr_psf_polar_fft

    # img_polar = fft.irfft(img_polar_fft, dim=0)
    # img = polar_transform.polar2img(img_polar, obj.shape)

    return img_polar_fft


def ring_convolve_batch(
    obj,
    seidel_coeffs,
    sys_params={},
    device=torch.device("cpu"),
    verbose=False,
):
    """

    Returns the ring convolution of an object with a stack of PSFs.

    Parameters
    ----------
    obj : torch.Tensor
        The image to be convolved with the PSF stack. Must be (N,N).

    seidel_coeffs : torch.Tensor
        Seidel aberration coefficients of the system. Should be (6, 1).

    device : torch.device, optional
        The device to use for the computation.

    verbose : bool, optional
        Whether to display a progress bar.

    Returns
    -------
    img : torch.Tensor
        The ring convolution of the object with the PSF stack. Will be (N,N).

    """

    if not torch.is_tensor(obj):
        obj = torch.tensor(obj).float()
    if obj.device is not device:
        obj = obj.to(device)

    # infer info from the object
    dim = obj.shape[0]

    def_sys_params = {
        "samples": dim,
        "L": 1e-3,
        "lamb": 0.55e-6,
        "pupil_radius": ((dim) * (0.55e-6) * (100e-3)) / (4 * (1e-3)),
        "z": 100e-3,
    }
    def_sys_params.update(sys_params)

    # get object RoFT
    obj_polar = polar_transform.img2polar(obj, numRadii=dim)

    rs = np.linspace(0, (dim / 2), dim, endpoint=False, retstep=False)
    point_list = [(r, -r) for r in rs]  # radial line of PSFs

    r_list = np.sqrt(2) * (
        np.linspace(0, (obj.shape[0] / 2), dim, endpoint=False, retstep=False) + 0.5
    )

    dr = r_list[1] - r_list[0]
    dtheta = 2 * np.pi / obj_polar.shape[0]

    obj_fft = fft.rfft(obj_polar, dim=0)

    # create blank image RoFT which will be subsequently filled in
    img_polar_fft = torch.zeros_like(obj_fft, dtype=torch.complex64, device=device)
    radii = tqdm(enumerate(r_list)) if verbose else (enumerate(r_list))
    for index, r in radii:
        psf_roft = seidel.compute_psfs(
            seidel_coeffs,
            [point_list[index]],
            sys_params=def_sys_params,
            polar=True,
            stack=False,
            buffer=0,
            verbose=False,
            device=device,
        )[0]
        psf_roft = torch.fft.rfft(psf_roft, dim=0)
        integration_area = r * dr * dtheta
        img_polar_fft += integration_area * (obj_fft[:, index][:, None] * psf_roft)
    img_polar = fft.irfft(img_polar_fft, dim=0)
    img = polar_transform.polar2img(img_polar, obj.shape)

    return img


def batch_ring_convolve(obj, psf_roft, device=torch.device("cpu")):
    """

    Returns the ring convolution of a batch of multichannel objects with a stack of PSFs.

    Parameters
    ----------
    obj : torch.Tensor
        The image to be convolved with the PSF stack. Must be (B,C,N,N) or (B,Z,C,N,N).

    psf_roft : torch.Tensor
        The stack of PSFs to convolve the image with. The PSFs should be in the
        Rotational Fourier domain. Should be (N, M, L) where N is the number of PSFs,
        M is the number of angles, and L is the number of radii the in the RoFT.

    device : torch.device, optional
        The device to use for the computation.

    Returns
    -------
    img : torch.Tensor
        The ring convolution of the object with the PSF stack. Will be (B,C,N,N).

    """
    if not torch.is_tensor(obj):
        obj = torch.tensor(obj)  # .float()
    if obj.device is not device:
        obj = obj.to(device)

    # infer info from the PSF roft
    num_radii = psf_roft.shape[0]

    # get object RoFT
    obj_polar = polar_transform.batchimg2polar(obj, numRadii=num_radii)
    obj_fft = fft.rfft(obj_polar, dim=-2)

    r_list = torch.tensor(
        np.sqrt(2)
        * (np.linspace(0, (obj.shape[-2] / 2), num_radii, endpoint=False) + 0.5),
        device=device,
    ).type(torch.complex64)
    dr = r_list[1] - r_list[0]
    dtheta = 2 * np.pi / obj_polar.shape[-2]

    # utilize einsum to accelerate batched multiplication
    rdrdtheta = r_list[None, None, :, None, None] * dr * dtheta
    # psf_roft = (psf_roft[None, None, ...] * rdrdtheta).type(torch.complex64)
    # img_polar_fft = obj_fft[:, :, None, :, :]

    img_polar_fft = torch.einsum(
        "bcwtr,bcrtw->bctw",
        obj_fft[:, :, None, :, :],
        (psf_roft[None, None, ...]) * rdrdtheta,
    )
    img = polar_transform.batchpolar2img(fft.irfft(img_polar_fft, dim=-2), obj.shape)

    return img


def convolve(obj, psf):
    """

    Returns the convolution of an object with a PSF.

    Parameters
    ----------
    obj : torch.Tensor
        The image to be convolved with the PSF. Must be (N,N).

    psf : torch.Tensor
        The PSF to convolve the image with. Must be (N,N).

    Returns
    -------
    img : torch.Tensor

    """

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


def full(obj, seidel_coeffs, sys_params={}, verbose=False, device=torch.device("cpu")):
    """

    Returns the full blur of an object with a stack of PSFs.

    Parameters
    ----------

    obj : np.ndarray
        The image to be convolved with the PSF stack. Must be (N,N).

    seidel_coeffs : np.ndarray
        The Seidel aberration coefficients to use for the PSF stack. Must be (6,1).

    sys_params : dict, optional
        The system parameters to use PSF generation, see `seidel.py` for details

    verbose : bool, optional
        Whether to display a progress bar.

    device : torch.device, optional
        The device to use for the computation.

    Returns
    -------
    img : torch.Tensor
        The full blur of the object with the PSF stack. Will be (N,N).

    """

    obj = torch.from_numpy(obj).float().to(device)

    def_sys_params = {
        "samples": obj.shape[0],
        "L": 1e-3,
        "lamb": 0.55e-6,
        "pupil_radius": ((obj.shape[0]) * (0.55e-6) * (100e-3)) / (4 * (1e-3)),
        "z": 100e-3,
    }
    def_sys_params.update(sys_params)

    seidel_coeffs = torch.tensor(seidel_coeffs)

    obj_shape = obj.shape
    img = torch.zeros(obj_shape, device=device)
    x_grid = np.linspace(-1, 1, obj_shape[1], endpoint=False) * obj_shape[1] / 2
    y_grid = np.linspace(1, -1, obj_shape[0], endpoint=False) * obj_shape[0] / 2

    if verbose:
        iterable_outer_loop = tqdm(np.arange(obj_shape[0]))
    else:
        iterable_outer_loop = np.arange(obj_shape[0])
    for i in iterable_outer_loop:
        for j in np.arange(obj_shape[1]):
            if obj[i, j] != 0:
                psf = seidel.compute_psfs(
                    seidel_coeffs,
                    [(x_grid[j], y_grid[i])],
                    sys_params=def_sys_params,
                    polar=False,
                    device=device,
                )[0]
                img += obj[i, j] * psf

    return img
