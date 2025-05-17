#!/usr/bin/env python
"""
This file is a slightly modified version of the original code: https://github.com/MicroscPSF/MicroscPSF-Py

Further details about the methodology can be found in there.
"""
import math
import torch
import torch.nn.functional as F


# Internal constants.
num_basis = 100  # Number of rescaled Bessels that approximate the phase function.
rho_samples = 1000  # Number of pupil sample along the radial direction.


def differentiable_arange(start, end, step, device):
    """
    Create a differentiable arange tensor.

    Parameters
    ----------
    start : float
        Start value.

    end : float
        End value.

    step : float
        Step size.

    device : torch.device
        Device to create the tensor on.

    Returns
    -------
    torch.Tensor
        A tensor with values from start to end with the specified step size.
    """
    num_elements = int((end - start) / step)
    return start + torch.arange(num_elements, device=device) * step


def calcRv_torch(dxy, xy_size, sampling=2, device=torch.device("cpu")):
    """
    Calculate rv vector, this is 2x up-sampled.

    Parameters
    ----------
    dxy : float
        Step size in the XY plane.

    xy_size : int
        Number of pixels in X/Y.

    sampling : int
        Sampling factor for the rv vector.

    device : torch.device
        Device to create the tensor on.

    Returns
    -------
    torch.Tensor
        A tensor with values from 0 to rv_max with the specified step size.
    """
    rv_max = math.sqrt(0.5 * xy_size * xy_size) + 1

    return differentiable_arange(0, rv_max * dxy, dxy / sampling, device=device)


def configure_torch(mp, wvl, device=torch.device("cpu")):
    """
    Configure the microscope parameters.

    Parameters
    ----------
    mp : dict
        The microscope parameters dictionary.

    wvl : float
        Light wavelength in microns.

    Returns
    -------
    list
        A list containing the Fourier-Bessel scaling factors and max_rho.
    """
    # Scaling factors for the Fourier-Bessel series expansion
    min_wavelength = 0.436  # microns
    scaling_factor = (
        mp["NA"]
        * (3 * torch.arange(1, num_basis + 1, device=device) - 2)
        * min_wavelength
        / wvl
    )

    # Not sure this is completely correct for the case where the axial
    # location of the flourophore is 0.0.
    #
    max_rho = (
        min([mp["NA"], mp["ng0"], mp["ng"], mp["ni0"], mp["ni"], mp["ns"]]) / mp["NA"]
    )

    return [scaling_factor, max_rho]


def deltaFocus(mp, zd):
    """
    Return focal offset needed to compensate for the camera being at zd.

    Parameters
    ----------
    mp : dict
        The microscope parameters dictionary.

    zd : float
        Actual camera position in microns.

    Returns
    -------
    float
        Focal offset needed to compensate for the camera being at zd.
    """

    a = mp["NA"] * mp["zd0"] / mp["M"]  # Aperture radius at the back focal plane.
    return a * a * (mp["zd0"] - zd) / (2.0 * mp["zd0"] * zd)


def gLXYZFocalScan_torch(
    mp,
    dxy,
    xy_size,
    zv,
    normalize=True,
    pz=0.0,
    wvl=0.6,
    zd=None,
    device=torch.device("cpu"),
):
    """
    Calculate 3D G-L PSF. This is models the PSF you would measure by scanning the microscopes
    focus.

    This will return a torch tensor with of size (zv.size, xy_size, xy_size). Note that z
    is the zeroth dimension of the PSF.

    Parameters
    ----------
    mp : dict
        The microscope parameters dictionary.

    dxy : float
        Step size in the XY plane.

    xy_size : int
        Number of pixels in X/Y.

    zv : torch.Tensor
        A torch tensor containing the (relative) z offset values of the coverslip (negative is
        closer to the objective).

    normalize : bool
        Normalize the PSF to unit height.

    pz : float
        Particle z position above the coverslip (positive values only).

    wvl : float
        Light wavelength in microns.

    zd : float
        Actual camera position in microns. If not specified the microscope tube length is used.

    device : torch.device
        Device to create the tensor on. Default is torch.device("cpu").

    Returns
    -------
    torch.Tensor
        A tensor with the G-L PSF of size (zv.size, xy_size, xy_size).
    """

    # Calculate rv vector, this is 2x up-sampled.
    rv = calcRv_torch(dxy, xy_size, device=device)

    # Calculate radial/Z PSF.
    PSF_rz = gLZRFocalScan_torch(
        mp, rv, zv, normalize=normalize, pz=pz, wvl=wvl, zd=zd, device=device
    )

    # Create XYZ PSF by interpolation.
    return psfRZToPSFXYZ_torch(dxy, xy_size, rv, PSF_rz, device=device)


def gLZRScan_torch(
    mp, pz, rv, zd, zv, normalize=True, wvl=0.6, device=torch.device("cpu")
):
    """
    Calculate radial G-L at specified radius.

    This function is primarily designed for internal use. Note that only one pz, zd and zv should be a torch tensor
    with more than one element. You can simulate scanning the focus, the particle
    or the camera but not 2 or 3 of these values at the same time.

    Parameters
    ----------
    mp : dict
        The microscope parameters dictionary.

    pz : torch.Tensor
        A torch tensor containing the particle z position above the coverslip (positive values only).

    rv : torch.Tensor
        A torch tensor containing the radius values.

    zd : torch.Tensor
        A torch tensor containing the actual camera position in microns.

    zv : torch.Tensor
        A torch tensor containing the relative z offset value of the coverslip (negative is
        closer to the objective).

    normalize : bool
        Normalize the PSF to unit height.

    wvl : float
        Light wavelength in microns.

    device : torch.device
        Device to create the tensor on. Default is torch.device("cpu").

    Returns
    -------
    torch.Tensor
        A tensor with the G-L PSF of size (zv.size, rv.size).
    """

    [scaling_factor, max_rho] = configure_torch(mp, wvl, device=device)
    rho = torch.linspace(0.0, max_rho, rho_samples, device=device)

    a = (
        mp["NA"] * mp["zd0"] / torch.sqrt(mp["M"] * mp["M"] + mp["NA"] * mp["NA"])
    )  # Aperture radius at the back focal plane.
    k = 2.0 * math.pi / wvl

    ti = zv.reshape(-1, 1) + mp["ti0"]
    pz = pz.reshape(-1, 1)
    zd = zd.reshape(-1, 1)

    opdt = OPD_torch(mp, rho, ti, pz, wvl, zd, device=device)

    # Sample the phase
    # phase = numpy.cos(opdt) + 1j * numpy.sin(opdt)
    phase = torch.exp(1j * opdt)

    # Define the basis of Bessel functions
    # Shape is (number of basis functions by number of rho samples)
    J = torch.special.bessel_j0(scaling_factor.reshape(-1, 1) * rho).type(
        torch.complex64
    )

    # Compute the approximation to the sampled pupil phase by finding the least squares
    # solution to the complex coefficients of the Fourier-Bessel expansion.
    # Shape of C is (number of basis functions by number of z samples).
    # Note the matrix transposes to get the dimensions correct.
    C, residuals, _, _ = torch.linalg.lstsq(J.T, phase.T)

    rv = rv * mp["M"]
    b = k * a * rv.reshape(-1, 1) / zd

    # Convenience functions for J0 and J1 Bessel functions
    J0 = lambda x: torch.special.bessel_j0(x)
    J1 = lambda x: torch.special.bessel_j1(x)

    # See equation 5 in Li, Xue, and Blu
    denom = scaling_factor * scaling_factor - b * b
    R = (
        scaling_factor * J1(scaling_factor * max_rho) * J0(b * max_rho) * max_rho
        - b * J0(scaling_factor * max_rho) * J1(b * max_rho) * max_rho
    ).type(torch.complex64)
    R /= denom

    # The transpose places the axial direction along the first dimension of the array, i.e. rows
    # This is only for convenience.
    PSF_rz = (torch.abs(R.mm(C)) ** 2).T

    # Normalize to the maximum value
    if normalize:
        PSF_rz.data = PSF_rz / PSF_rz.sum()

    return PSF_rz


def gLZRFocalScan_torch(
    mp, rv, zv, normalize=True, pz=0.0, wvl=0.6, zd=None, device=torch.device("cpu")
):
    """
    Calculate radial G-L at specified radius and z values.

    This models the PSF you would measure by scanning the microscope's focus.

    Parameters
    ----------
    mp : dict
        The microscope parameters dictionary.

    rv : torch.Tensor
        A torch tensor containing the radius values.

    zv : torch.Tensor
        A torch tensor containing the (relative) z offset values of the coverslip (negative is
        closer to the objective).

    normalize : bool
        Normalize the PSF to unit height.

    pz : float
        Particle z position above the coverslip (positive values only).

    wvl : float
        Light wavelength in microns.

    zd : float
        Actual camera position in microns. If not specified the microscope tube length is used.

    device : torch.device
        Device to create the tensor on. Default is torch.device("cpu").

    Returns
    -------
    PSF_rz : torch.Tensor
        A tensor with the G-L PSF of size (zv.size, rv.size).
    """

    if zd is None:
        zd = mp["zd0"]

    pz = torch.tensor([pz], device=device)
    zd = torch.tensor([zd], device=device)

    return gLZRScan_torch(
        mp, pz, rv, zd, zv, normalize=normalize, wvl=wvl, device=device
    )


def OPD_torch(mp, rho, ti, pz, wvl, zd, device=torch.device("cpu")):
    """
    Calculate phase aberration term.

    This is the phase aberration term in the G-L PSF.

    Parameters
    ----------
    mp : dict
        The microscope parameters dictionary.

    rho : torch.tensor
        A torch tensor containing the rho values.

    ti : torch.Tensor
        A torch tensor containing the coverslip z offset in microns.

    pz : torch.Tensor
        A torch tensor containing the particle z position above the coverslip in microns.

    wvl : float
        Light wavelength in microns.

    zd : torch.Tensor
        A torch tensor containing the actual camera position in microns.

    device : torch.device
        Device to create the tensor on. Default is torch.device("cpu").

    Returns
    -------
    torch.tensor
        A torch tensor containing the phase aberration term.
    """

    NA = mp["NA"]
    ns = mp["ns"]
    ng0 = mp["ng0"]
    ng = mp["ng"]
    ni0 = mp["ni0"]
    ni = mp["ni"]
    ti0 = mp["ti0"]
    tg = mp["tg"]
    tg0 = mp["tg0"]
    zd0 = mp["zd0"]

    a = NA * zd0 / mp["M"]  # Aperture radius at the back focal plane.
    k = 2.0 * math.pi / wvl  # Wave number of emitted light.

    OPDs = pz * torch.sqrt(ns * ns - NA * NA * rho * rho)  # OPD in the sample.
    OPDi = ti * torch.sqrt(ni * ni - NA * NA * rho * rho) - ti0 * torch.sqrt(
        ni0 * ni0 - NA * NA * rho * rho
    )  # OPD in the immersion medium.
    OPDg = tg * torch.sqrt(ng * ng - NA * NA * rho * rho) - tg0 * torch.sqrt(
        ng0 * ng0 - NA * NA * rho * rho
    )  # OPD in the coverslip.
    OPDt = a * a * (zd0 - zd) * rho * rho / (2.0 * zd0 * zd)  # OPD in camera position.

    return k * (OPDs + OPDi + OPDg + OPDt)


def psfRZToPSFXYZ_torch(dxy, xy_size, rv, PSF_rz, device=torch.device("cpu")):
    """
    Use interpolation to create a 3D XYZ PSF from a 2D ZR PSF.

    Parameters
    ----------
    dxy : float
        Step size in the XY plane.

    xy_size : int
        Number of pixels in X/Y.

    rv : torch.Tensor
        A torch tensor containing the radius values.

    PSF_rz : torch.Tensor
        A tensor with the G-L PSF of size (zv.size, rv.size).

    device : torch.device
        Device to create the tensor on. Default is torch.device("cpu").

    Returns
    -------
    PSF_xyz : torch.Tensor
        A tensor with the G-L PSF of size (zv.size, xy_size, xy_size).
    """
    # Create XY grid of radius values.
    c_xy = float(xy_size) * 0.5
    grid = torch.arange(0, xy_size, device=device)
    xy = torch.stack(torch.meshgrid(grid, grid, indexing="xy")) + 0.5
    r_pixel = dxy * torch.sqrt(
        (xy[1] - c_xy) * (xy[1] - c_xy) + (xy[0] - c_xy) * (xy[0] - c_xy)
    )

    # Create XYZ PSF by interpolation.
    # PSF_xyz = numpy.zeros((PSF_rz.shape[0], xy_size, xy_size))
    # for i in range(PSF_rz.shape[0]):
    #     psf_rz_interp = scipy.interpolate.interp1d(rv, PSF_rz[i, :])
    #     PSF_xyz[i, :, :] = psf_rz_interp(r_pixel.ravel()).reshape(xy_size, xy_size)

    # PSF_rz is Z by R, we need to interpolate for each Z value so we will treate Z as the batch dim
    input = PSF_rz[None, :, :, None]
    rpix = (r_pixel / r_pixel.max() - 0.5) * 2

    grid = rpix[None, :]
    grid = torch.stack([-torch.ones_like(grid), grid], dim=-1)
    PSF_xyz = F.grid_sample(input, grid, align_corners=True).squeeze()

    return PSF_xyz
