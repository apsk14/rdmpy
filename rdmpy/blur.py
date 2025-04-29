"""Implementations of blur models"""

import numpy as np
import torch
import torch.fft as fft
import torch.nn.functional as F

from tqdm import tqdm

from ._src import polar_transform, seidel


def ring_convolve(
    obj,
    psf_roft,
    device=torch.device("cpu"),
    verbose=False,
):
    """Return the ring convolution of an object with a stack of PSFs.

    This function simulates the spatially-varying blur of a rotationally symmetric imaging system.
    The PSFs are assumed to be in the Rotational Fourier domain.

    Parameters
    ----------
    obj : np.ndarray or torch.Tensor
        The image to be convolved with the PSF stack. Must be (N,N) where N is even.

    psf_roft : torch.Tensor
        The stack of PSFs to convolve the image with. The PSFs should be in the
        Rotational Fourier domain. Should be (L, M, L) where L is the number of PSFs/radii and
        M is the number of angles. See `rdmpy.calibrate.get_psfs` for details.

    device : torch.device
        The device to use for the computation.

    verbose : bool
        Whether to display a progress bar.

    Returns
    -------
    img : torch.Tensor
        The ring convolution of the object with the PSF stack. Will be (N,N).

    Notes
    -----
    This function assumes that the PSFs are in the Rotational Fourier domain.
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


def sheet_convolve(obj, psf_stack, device=torch.device("cpu")):
    r"""Return the slab convolution of an image stack with a stack of PSFs.

    This function simulates the spatially-varying blur of a light-sheet microscope, where
    the light-sheet defocuses away from the center. The PSFs will be a list of 3D PSFs moving
    from center to the one edge of the light-sheet. We assume that the light-sheet is a gaussian
    beam that defocuses the same both sides from the center.

    Parameters
    ----------
    obj : np.ndarray or torch.Tensor
        The obj stack to be convolved with the PSF stack. Must be (M,N,T).

    psf_stack : torch.Tensor
        The stack of PSFs to convolve the image with. Must be a list of 3D PSFs of length
        M//2 (half of the varying axis of the obj).

    device : torch.device
        The device to use for the computation.

    Returns
    -------
    img : torch.Tensor
        The slab convolution of the image stack with the PSF stack. Will be (M,N,T).
    """

    if not torch.is_tensor(obj):
        obj = torch.tensor(obj).float()
    if obj.device is not device:
        obj = obj.to(device)

    result_tensor = torch.zeros_like(obj).to(device)
    for i in range(obj.shape[0]):
        curr_img_slice = obj[i, :, :].T
        curr_img_slice = curr_img_slice[None, None, :, :]
        if i < obj.shape[0] // 2:
            curr_psf = psf_stack[obj.shape[0] // 2 - i - 1]
        else:
            curr_psf = psf_stack[i - obj.shape[0] // 2]

        output_chunk = F.conv2d(
            curr_img_slice,
            torch.flip(curr_psf[None, :].permute(1, 0, 3, 2), [2, 3]),
            padding=(curr_psf.shape[2] // 2, curr_psf.shape[1] // 2),
        ).squeeze()

        if output_chunk.shape[1] != curr_img_slice.shape[2]:
            output_chunk = output_chunk[:, 1:, :]
        if output_chunk.shape[2] != curr_img_slice.shape[3]:
            output_chunk = output_chunk[:, :, 1:]
        if (
            i >= output_chunk.shape[0] // 2
            and i <= obj.shape[0] - output_chunk.shape[0] // 2
        ):
            result_tensor[
                i - output_chunk.shape[0] // 2 : i + output_chunk.shape[0] // 2, :, :
            ] += output_chunk.permute(0, 2, 1)
        elif i < output_chunk.shape[0] // 2:
            result_tensor[
                0 : i + output_chunk.shape[0] // 2, :, :
            ] += output_chunk.permute(0, 2, 1)[output_chunk.shape[0] // 2 - i :, :, :]
        else:
            result_tensor[
                i - output_chunk.shape[0] // 2 :, :, :
            ] += output_chunk.permute(0, 2, 1)[
                0 : obj.shape[0] - (output_chunk.shape[0] // 2 + i), :, :
            ]

    return result_tensor


# TODO: Under production
# def batch_ring_convolve(obj, psf_roft, device=torch.device("cpu")):
#     """Return the ring convolution of a batch of objects with a single stack of PSFs.

#     Parameters
#     ----------
#     obj : np.ndarray or torch.Tensor
#         The image to be convolved with the PSF stack. Must be (B,N,N) where N is even.

#     psf_roft : torch.Tensor
#         The stack of PSFs to convolve the image with. The PSFs should be in the Rotational Fourier domain.
#         Should be (L, M, L) where L is the number of PSFs/radii and M is the number of angles.

#     device : torch.device,
#         The device to use for the computation.

#     Returns
#     -------
#     img : torch.Tensor
#         The ring convolution of the object with the PSF stack. Will be (B,N,N).
#     """

#     if not torch.is_tensor(obj):
#         obj = torch.tensor(obj)  # .float()
#     if obj.device is not device:
#         obj = obj.to(device)

#     # infer info from the PSF roft
#     num_radii = psf_roft.shape[0]

#     # get object RoFT
#     obj_polar = polar_transform.batchimg2polar(obj, numRadii=num_radii)
#     obj_fft = fft.rfft(obj_polar, dim=-2)

#     r_list = torch.tensor(
#         np.sqrt(2)
#         * (np.linspace(0, (obj.shape[-2] / 2), num_radii, endpoint=False) + 0.5),
#         device=device,
#     ).type(torch.complex64)
#     dr = r_list[1] - r_list[0]
#     dtheta = 2 * np.pi / obj_polar.shape[-2]

#     # utilize einsum to accelerate batched multiplication
#     rdrdtheta = r_list[None, None, :, None, None] * dr * dtheta
#     # psf_roft = (psf_roft[None, None, ...] * rdrdtheta).type(torch.complex64)
#     # img_polar_fft = obj_fft[:, :, None, :, :]

#     img_polar_fft = torch.einsum(
#         "bcwtr,bcrtw->bctw",
#         obj_fft[:, :, None, :, :],
#         (psf_roft[None, None, ...]) * rdrdtheta,
#     )
#     img = polar_transform.batchpolar2img(fft.irfft(img_polar_fft, dim=-2), obj.shape)

#     return img


def convolve(obj, psf):
    """Return the convolution of an object with a PSF.

    Parameters
    ----------
    obj : np.ndarray or torch.Tensor
        The image to be convolved with the PSF. Must be (N,N).

    psf : torch.Tensor
        The PSF to convolve the image with. Must be (N,N).

    Returns
    -------
    img : torch.Tensor
        The convolution of the object with the PSF. Will be (N,N).

    Notes
    -----
    Here we use the real Fourier transform for efficiency. Thus the object and PSF must both be real-valued.
    """

    extent = obj.shape[0]
    padded_obj = F.pad(obj, (0, extent, 0, extent))
    padded_psf = F.pad(psf, (0, extent, 0, extent))
    padded_img = fft.irfftn(fft.rfftn(padded_obj) * fft.rfftn(padded_psf))
    padded_img = torch.roll(
        padded_img,
        shifts=(-padded_img.shape[0] // 4, -padded_img.shape[1] // 4),
        dims=(0, 1),
    )

    return padded_img[0:-extent, 0:-extent]


def full(obj, seidel_coeffs, sys_params={}, verbose=False, device=torch.device("cpu")):
    """Return the full blur of an object using Seidel coefficients.

    This is the linear forward model with no assumptions on the PSFs. Each PSF in the field-of-view is manually superimposed
    and weighted by the object. This is a slow process, but it is the most accurate way to simulate the spatially-varying blur of an object.
    Since it is generally not possible to store all of the PSFs in GPU memory, this function will compute the PSFs on-the-fly using Seidel coefficients.

    Parameters
    ----------
    obj : np.ndarray or torch.Tensor
        The image to be convolved with the PSF stack. Must be (N,N).

    seidel_coeffs : np.ndarray
        The Seidel aberration coefficients to use for the PSF stack. Must be (6,1).

    sys_params : dict,
        The system parameters to use PSF generation, see `seidel.py` for details

    verbose : bool,
        Whether to display a progress bar.

    device : torch.device,
        The device to use for the computation.

    Returns
    -------
    img : torch.Tensor
        The full blur of the object with the PSF stack. Will be (N,N).
    """

    if not torch.is_tensor(obj):
        obj = torch.tensor(obj)  # .float()
    if obj.device is not device:
        obj = obj.to(device)

    def_sys_params = {
        "samples": obj.shape[0],
        "L": 0,
        "lamb": 0.55e-6,
        "NA": 0.5,
    }
    radius_over_z = np.tan(np.arcsin(def_sys_params["NA"]))
    def_sys_params["L"] = ((obj.shape[0]) * (def_sys_params["lamb"])) / (
        4 * (radius_over_z)
    )
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
