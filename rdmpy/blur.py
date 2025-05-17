"""Implementations of blur models"""

import numpy as np
import torch
import torch.fft as fft
import torch.nn.functional as F

from tqdm import tqdm
from torch.utils.checkpoint import checkpoint

from ._src import polar_transform, psf_model


def ring_convolve(obj, psf_data, patch_size=0, device=torch.device("cpu")):
    """Return the ring convolution of an object with a stack of PSFs.

    This function simulates the spatially-varying blur of a rotationally symmetric imaging system.
    The PSFs are assumed to be in the Rotational Fourier domain.

    Parameters
    ----------
    obj : np.ndarray or torch.Tensor
        The image to be convolved with the PSF stack. Must be (N,N) where N is even.

    psf_data : torch.Tensor
        The stack of PSFs to convolve the image with. The PSFs should be in the Rotational Fourier domain.
        Should be (L, M, L) where L is the number of PSFs/radii and M is the number of angles.

    patch_size : int
        The size of isoplanatic annuli. If 0, will compute ring by ring.

    device : torch.device
        The device to use for the computation.

    Returns
    -------
    img : torch.Tensor
        The ring convolution of the object with the PSF stack. Will be (N,N).
    """

    if not torch.is_tensor(obj):
        obj = torch.tensor(obj).float()
    obj = obj.to(device)
    if not len(obj.shape) == 2 or obj.shape[0] != obj.shape[1]:
        raise AssertionError(f"Object of shape {obj.shape} must be 2D square")

    if patch_size < 0 or patch_size > obj.shape[0]:
        raise AssertionError(
            f"Patch size {patch_size} must be between 0 and {obj.shape[0]}"
        )

    if patch_size == 0:
        return ring_convolve_full(obj, psf_data, device=device)
    else:
        return ring_convolve_patch(obj, psf_data, patch_size, device=device)


def ring_convolve_patch(
    obj,
    psf_stack,
    patch_size,
    device=torch.device("cpu"),
):

    pad = obj.shape[0] // 2
    overlap = patch_size // 4
    num_rings = obj.shape[0]
    num_patches = int(np.ceil(num_rings / patch_size))

    if num_patches != psf_stack.shape[0]:
        raise AssertionError(
            f"Number of patches {num_patches} does not match number of PSFs {psf_stack.shape[0]}"
        )

    # Polar transform (tracked for gradient)
    obj_polar = polar_transform.batchimg2polar(
        obj[None, None, :, :], numRadii=num_rings
    ).squeeze()

    # Build masks outside autograd
    with torch.no_grad():
        masks = []
        taper = (
            torch.linspace(0.0, 1.0, overlap, device=device) if overlap > 0 else None
        )
        for i in range(num_patches):
            m = torch.zeros_like(obj_polar, device=device)
            start = i * patch_size
            end = (i + 1) * patch_size
            m[:, start:end] = 1
            if overlap > 0:
                if i > 0:
                    m[:, start : start + overlap] = taper
                if end + overlap < num_rings:
                    m[:, end : end + overlap] = 1 - taper
                elif end < num_rings:
                    short_taper = torch.linspace(
                        0.0, 1.0, num_rings - end, device=device
                    )
                    m[:, end:] = 1 - short_taper
            masks.append(m)

    # Forward patch function (wrapped for checkpointing)
    def process_patch(masked_patch, psf_fft):
        recon = polar_transform.batchpolar2img(
            masked_patch[None, None, :, :], obj.shape
        ).squeeze()
        padded = F.pad(recon, (0, pad, 0, pad))
        return psf_fft * fft.rfft2(padded)

    # Accumulate result
    result = torch.zeros(
        [psf_stack.shape[1], psf_stack.shape[2] // 2],
        device=device,
        dtype=torch.complex64,
    )
    for i in range(num_patches):
        # Masked polar patch (this part tracks gradients)
        masked = obj_polar * masks[i]

        # checkpointed computation to reduce memory use
        result += checkpoint(
            process_patch,
            masked,
            psf_stack[i, :, 0 : psf_stack.shape[-1] // 2]
            + 1j * psf_stack[i, :, psf_stack.shape[-1] // 2 :],
        )

    result = fft.irfft2(result)

    # Centering and crop
    result = torch.roll(result, shifts=(-pad, -pad), dims=(-2, -1))
    return result[:-pad, :-pad]


def ring_convolve_full(
    obj,
    psf_roft,
    device=torch.device("cpu"),
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

    Returns
    -------
    img : torch.Tensor
        The ring convolution of the object with the PSF stack. Will be (N,N).

    Notes
    -----
    This function assumes that the PSFs are in the Rotational Fourier domain.
    """

    # infer info from the PSF roft
    num_radii = psf_roft.shape[0]

    # get object RoFT
    obj_polar = polar_transform.img2polar(obj, numRadii=obj.shape[0])

    r_list = np.sqrt(2) * (
        np.linspace(0, (obj.shape[0] / 2), obj.shape[0], endpoint=False, retstep=False)
        + 0.5
    )

    dr = r_list[1] - r_list[0]
    dtheta = 2 * np.pi / psf_roft.shape[1]
    obj_fft = fft.rfft(obj_polar, dim=0)

    # create blank image RoFT which will be subsequently filled in
    img_polar_fft = torch.zeros_like(obj_fft, dtype=torch.complex64, device=device)

    for index in torch.arange(num_radii):
        integration_area = r_list[index] * dr * dtheta
        curr_psf_polar_fft = (
            psf_roft[index, 0 : psf_roft.shape[1] // 2, :]
            + 1j * psf_roft[index, psf_roft.shape[1] // 2 :, :]
        )
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

    if not torch.is_tensor(obj):
        obj = torch.tensor(obj).float()
    if not len(obj.shape) == 2 and obj.shape[0] == obj.shape[1]:
        raise AssertionError(f"Object of shape {obj.shape} must be 2d square image")

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
        The system parameters to use PSF generation, see `psf_model.py` for details

    verbose : bool,
        Whether to display a progress bar.

    device : torch.device,
        The device to use for the computation.

    Returns
    -------
    img : torch.Tensor
        The full blur of the object with the PSF stack. Will be (N,N).

    Notes
    -----
    This function is extremely slow and is primarily for comparison purposes.
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
                psf = psf_model.compute_rdm_psfs(
                    seidel_coeffs,
                    [(x_grid[j], y_grid[i])],
                    sys_params=def_sys_params,
                    polar=False,
                    device=device,
                )[0]
                img += obj[i, j] * psf

    return img
