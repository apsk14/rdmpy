"""
This file contains functions for computing PSFs from seidel coefficients.
"""

import pathlib
import pdb

import numpy as np
import torch
from torch.nn.functional import interpolate

import matplotlib as mpl
from tqdm import tqdm

from . import gibson_lanni as gl, util, polar_transform

dirname = str(pathlib.Path(__file__).parent.parent.absolute())
mpl.rcParams["figure.dpi"] = 500


# Note coeffs expected in following order: Wd, W040, W131, W222, W220, W311
def compute_pupil_phase(coeffs, X, Y, u, v, higher_order=None):
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

    higher_order : torch.Tensor, optional
       6th order coefficients of the optical system.


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

    if higher_order is not None:
        higher_order_pupil = (
            higher_order[0] * pupil_radii**3
            + higher_order[1] * obj_rad * pupil_radii**2 * X_rot
            + higher_order[2] * obj_rad**2 * pupil_radii * X_rot**2
            + higher_order[3] * obj_rad**2 * pupil_radii**2
            + higher_order[4] * obj_rad**3 * pupil_radii * X_rot
            + higher_order[5] * obj_rad**4 * X_rot**2
            + higher_order[6] * obj_rad**4 * pupil_radii
            + higher_order[7] * obj_rad**5 * X_rot
        )
        pupil_phase += higher_order_pupil

    return pupil_phase


def compute_rdm_psfs(
    coeffs,
    desired_list,
    dim=None,
    sys_params={},
    polar=False,
    stack=False,
    buffer=2,
    shift=True,
    downsample=1,
    higher_order=None,
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

    buffer : int, optional
        How many extra rows to add to each PSF for RoFT.

    downsample : int, optional
        Factor by which to downsample the PSFs.

    higher_order : torch.Tensor, optional
        6th order coefficients of the optical system.

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
            "L": 0,
            "lamb": 0.55e-6,
            "NA": 0.5,
        }
        radius_over_z = np.tan(np.arcsin(def_sys_params["NA"]))
        def_sys_params["L"] = ((dim) * (def_sys_params["lamb"])) / (4 * (radius_over_z))

        def_sys_params.update(sys_params)
        sys_params = def_sys_params
    # num_radii = len(desired_list)
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
    radius_over_z = np.tan(np.arcsin(sys_params["NA"]))
    k = (2 * np.pi) / lamb
    fx = torch.linspace(-1 / (2 * dt), 1 / (2 * dt), samples, device=device)
    [Fx, Fy] = torch.meshgrid((fx, fx), indexing="xy")
    scale_factor = lamb / radius_over_z
    circle = circ(
        torch.sqrt(torch.square(Fx) + torch.square(Fy)) * scale_factor, radius=1
    )
    if stack:
        if polar:
            if samples > 700:
                desired_psfs = torch.zeros(
                    (
                        len(desired_list),
                        int(samples // downsample) * 4 + buffer,
                        int(samples // downsample),
                    ),
                    device=device,
                )  # add two extra rows for RoFT later
            else:
                desired_psfs = torch.zeros(
                    (
                        len(desired_list),
                        int(samples // downsample) * 4 + buffer,
                        int(samples // downsample),
                    ),
                    device=device,
                )  # add two extra rows for RoFT later
        else:
            desired_psfs = torch.zeros(
                (
                    len(desired_list),
                    int(samples // downsample),
                    int(samples // downsample) + buffer,
                ),
                device=device,
            )
    else:
        desired_psfs = []
    idx = 0
    if higher_order is not None:
        higher_order = lamb * higher_order
    for point in iterable_coords:

        W = compute_pupil_phase(
            lamb * coeffs,
            X=-Fx * scale_factor,
            Y=-Fy * scale_factor,
            u=(point[0]) / (samples / 2),
            v=-(point[1]) / (samples / 2),
            higher_order=higher_order,
        )
        H = circle * torch.exp(-1j * k * W)
        H[circle < 1e-12] = 0
        curr_psf = torch.fft.fftshift(torch.fft.ifftn(torch.fft.ifftshift(H)))
        del H
        curr_psf = torch.square(torch.abs(curr_psf))

        curr_psf = curr_psf / curr_psf.sum()

        if shift:
            curr_psf = util.shift_torch(
                curr_psf,
                (-(point[1].cpu().numpy()), (point[0].cpu().numpy())),
                mode="bilinear",
            )

        if downsample != 1:
            curr_psf = interpolate(
                curr_psf.unsqueeze(0).unsqueeze(0),
                scale_factor=1 / downsample,
                mode="bilinear",
            ).squeeze()

        if polar:
            curr_psf = polar_transform.img2polar(
                curr_psf.float(), numRadii=int(dim // downsample)
            )
        if stack:
            if buffer > 0:
                if polar:
                    desired_psfs[idx, :-buffer, :] = (
                        curr_psf  # Leaving two extra rows for RoFT later
                    )
                else:
                    desired_psfs[idx, :, :-buffer] = curr_psf
            else:
                desired_psfs[idx] = curr_psf
        else:
            desired_psfs += [curr_psf]
        idx += 1

    return desired_psfs


def compute_sdm_psfs(
    waist,
    spread,
    gl_params,
    num_psfs,
    xy_dim,
    zmax,
    oversample=2,
    verbose=True,
    device=torch.device("cpu"),
):

    psf_cube_list = []
    z = (
        torch.linspace(-(zmax // 2), zmax // 2, oversample * zmax, device=device)
        * gl_params["dz"]
    )
    gl_psf = gl.gLXYZFocalScan_torch(
        gl_params,
        dxy=gl_params["dx"] / oversample,
        xy_size=oversample * xy_dim,
        zv=z,
        normalize=True,
        wvl=gl_params["wavelength"],
        device=device,
    ).permute(1, 2, 0)

    iterable_coords = tqdm(range(num_psfs)) if verbose else range(num_psfs)

    for i in iterable_coords:
        # where along the light sheet axis is the current psf
        lateral_pos = i * gl_params["dx"]
        curr_psf = compute_GL_psf(
            lateral_pos=lateral_pos,
            waist=waist,
            spread=spread,
            gl_params=gl_params,
            xy_dim=xy_dim,
            zmax=zmax,
            oversample=oversample,
            precomputed_psf=gl_psf,
            device=device,
        )
        psf_cube_list += [curr_psf]

    for i in range(len(psf_cube_list)):
        psf_cube_list[i] = psf_cube_list[i] / psf_cube_list[-1].sum()

    return psf_cube_list


def compute_GL_psf(
    lateral_pos,
    waist,
    spread,
    gl_params,
    xy_dim,
    zmax,
    oversample=2,
    precomputed_psf=None,
    device=torch.device("cpu"),
):
    # Get the axial spread and extent that lateral position
    sigma = waist * torch.sqrt(1 + ((lateral_pos) / spread) ** 2)
    if zmax % 2 != 0:
        zmax = zmax + 1
    z_dim = int(torch.floor((6 * sigma) / gl_params["dz"]))
    # round z_dim to nearest even number
    z_dim = z_dim + 1 if z_dim % 2 != 0 else z_dim
    if z_dim > zmax:
        z_dim = zmax

    # Get the z grid along which the PSF will be evaluated.
    z = (
        torch.linspace(-(z_dim // 2), z_dim // 2, oversample * z_dim, device=device)
        * gl_params["dz"]
    )

    if precomputed_psf is None:
        curr_psf = gl.gLXYZFocalScan_torch(
            gl_params,
            dxy=gl_params["dx"] / oversample,
            xy_size=oversample * xy_dim,
            zv=z,
            normalize=True,
            wvl=gl_params["wavelength"],
            device=device,
        ).permute(1, 2, 0)
    else:
        # trim both sides of precomputed psf so that it's last dimension is of length z_dim
        curr_psf = precomputed_psf[
            :,
            :,
            int(((precomputed_psf.shape[2] - (z_dim * oversample)) / 2)) : int(
                (precomputed_psf.shape[2] + (z_dim * oversample)) / 2
            ),
        ]

    z_full = (
        torch.linspace(-(zmax // 2), zmax // 2, oversample * zmax, device=device)
        * gl_params["dz"]
    )

    gaussian_weight = (waist / sigma) * torch.exp(
        -((z_full**2) / (2 * sigma**2))
    ).to(device)
    # gaussian_weight = gaussian_weight / gaussian_weight.sum()

    # truncate z full on either side so its the same length as z
    gaussian_weight = gaussian_weight[
        int((z_full.shape[0] - z.shape[0]) / 2) : int(
            (z_full.shape[0] + z.shape[0]) / 2
        )
    ]

    curr_psf = gaussian_weight * curr_psf

    # downsample by a factor of 2
    curr_psf = interpolate(
        curr_psf[None, None, :, :, :], scale_factor=1 / oversample, mode="trilinear"
    ).squeeze()

    return curr_psf


def make_gaussian(var, dim, L, device):
    """
    Returns a 2D Gaussian with variance `var` and sidelength `dim`.

    Parameters
    ----------
    var : float
        Variance of the desired Gaussian mask.

    dim : int
        Sidelength of the desired Gaussian mask.

    device : torch.device
        Device to run on.

    Returns
    -------
    mask : torch.Tensor
        2D Gaussian with variance `var` and sidelength `dim`.

    """
    x = torch.linspace(-L / 2, L / 2, dim + 1, device=device)[1:] * 1e6
    X, Y = torch.meshgrid(x, x, indexing="ij")
    kernel = torch.exp(-((X**2) / (2 * var[0]) + (Y**2) / (2 * var[1])))

    # pdb.set_trace()

    kernel = kernel / torch.sum(kernel)

    return kernel


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
