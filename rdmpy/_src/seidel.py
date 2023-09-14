"""
This file contains functions for computing PSFs from seidel coefficients.
"""

import pathlib

import numpy as np
import torch

import matplotlib as mpl
from tqdm import tqdm

from . import util, polar_transform

dirname = str(pathlib.Path(__file__).parent.parent.absolute())
mpl.rcParams["figure.dpi"] = 500


# Note coeffs expected in following order: Wd, W040, W131, W222, W220, W311
def compute_pupil_phase(coeffs, X, Y, u, v):
    """
    Computes the Seidel polynomial given a grid and coefficients

    Parameters
    ----------
    coeffs : torch.Tensor
        Seidel coefficients of the optical system.

    X : torch.Tensor
        X coordinates of the grid.

    Y : torch.Tensor
        Y coordinates of the grid.

    u : torch.Tensor
        X coordinate of the point in the object plane.

    v : torch.Tensor
        Y coordinate of the point in the object plane.

    Returns
    -------
    pupil_phase : torch.Tensor
        Phase of the pupil function at the given point.

    """

    #  first rotate grid as needed
    rot_angle = torch.atan2(v, u)
    obj_rad = torch.sqrt(u**2 + v**2)
    X_rot = X * torch.cos(rot_angle) + Y * torch.sin(rot_angle)
    Y_rot = -X * torch.sin(rot_angle) + Y * torch.cos(rot_angle)

    pupil_radii = torch.square(X_rot) + torch.square(Y_rot)
    pupil_phase = (
        coeffs[0] * torch.square(pupil_radii)
        + coeffs[1] * obj_rad * pupil_radii * X_rot
        + coeffs[2] * (obj_rad**2) * torch.square(X_rot)
        + coeffs[3] * (obj_rad**2) * pupil_radii
        + coeffs[4] * (obj_rad**3) * X_rot
        + coeffs[5] * pupil_radii
    )

    return pupil_phase


def compute_psfs(
    coeffs,
    desired_list,
    dim=None,
    sys_params={},
    polar=False,
    stack=False,
    verbose=False,
    device=torch.device("cpu"),
):
    """
    Parameters
    ----------
    coeffs : torch.Tensor
        Seidel coefficients of the optical system.

    desired_list : list
        List of desired points in the object plane to compute PSFs at. (x,y) format

    dim : int, optional
        Desired sidelength of each PSF image. Note that it enforces square images.

    sys_params : dict, optional
        Parameters for the optical system. See `seidel.py` for details.

    polar : bool, optional
        Whether to return PSFs in polar coordinates.

    stack : bool, optional
        Whether to stack PSFs into a single tensor.

    verbose : bool, optional
        Whether to display a progress bar.

    device : torch.device, optional
        Which device to run on.

    Returns
    -------
    desired_psfs : torch.Tensor
        PSFs of the optical system at locations specified by `desired_list`.

    """
    if dim is None and sys_params == {}:
        raise NotImplementedError
    if dim is None:
        dim = sys_params["samples"]
    else:
        def_sys_params = {
            "samples": dim,
            "L": 1e-3,
            "lamb": 0.55e-6,
            "pupil_radius": ((dim) * (0.55e-6) * (100e-3)) / (4 * (1e-3)),
            "z": 100e-3,
        }

        def_sys_params.update(sys_params)
        sys_params = def_sys_params
    num_radii = len(desired_list)
    desired_list = [
        (
            torch.tensor(i[0], device=device).float(),
            torch.tensor(i[1], device=device).float(),
        )
        for i in desired_list
    ]
    iterable_coords = tqdm(desired_list) if verbose else desired_list
    samples = sys_params["samples"]
    L = sys_params["L"]
    dt = L / samples
    lamb = sys_params["lamb"]
    k = (2 * np.pi) / lamb
    pupil_radius = sys_params["pupil_radius"]
    z = sys_params["z"]
    fx = np.linspace(-1 / (2 * dt), 1 / (2 * dt), samples)
    [Fx, Fy] = torch.tensor(np.meshgrid(fx, fx), device=device)
    scale_factor = (lamb * z) / pupil_radius
    circle = circ(
        torch.sqrt(torch.square(Fx) + torch.square(Fy)) * scale_factor, radius=1
    )
    if stack:
        if polar:
            if samples > 500:
                desired_psfs = torch.zeros(
                    (samples, samples * 2 + 2, samples),
                    device=device,
                )  # add two extra rows for RoFT later
            else:
                desired_psfs = torch.zeros(
                    (samples, samples * 4 + 2, samples), device=device
                )  # add two extra rows for RoFT later
        else:
            desired_psfs = torch.zeros((samples, samples, samples), device=device)
    else:
        desired_psfs = []
    idx = 0
    for point in iterable_coords:
        W = compute_pupil_phase(
            lamb * coeffs,
            X=-Fx * scale_factor,
            Y=-Fy * scale_factor,
            u=(point[0]) / (samples / 2),
            v=-(point[1]) / (samples / 2),
        )
        H = circle * torch.exp(-1j * k * W)
        H[circle < 1e-12] = 0
        curr_psf = torch.fft.fftshift(torch.fft.ifftn(torch.fft.ifftshift(H)))
        del H
        curr_psf = torch.square(torch.abs(curr_psf))
        curr_psf = curr_psf / curr_psf.sum()

        curr_psf = util.shift_torch(
            curr_psf,
            (-(point[1].cpu().numpy()), (point[0].cpu().numpy())),
            mode="bicubic",
        )
        if polar:
            curr_psf = polar_transform.img2polar(curr_psf.float(), numRadii=num_radii)
        if stack:
            # desired_psfs[idx] = curr_psf
            desired_psfs[
                idx, :-2, :
            ] = curr_psf  # Leaving two extra rows for RoFT later
        else:
            desired_psfs += [curr_psf]
        idx += 1

    return desired_psfs


def circ(r, radius):
    """
    Returns a circular mask of radius `radius` centered at the origin.

    Parameters
    ----------
    r : torch.Tensor
        Meshgrid of radii.

    radius : float
        Radius of the desired mask.

    Returns
    -------
    mask : torch.Tensor
        Circular mask of radius `radius` centered at the origin.

    """
    return abs(r) <= radius
