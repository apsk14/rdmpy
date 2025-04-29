#!/usr/bin/env python
"""
Code was modified from the original code: https://github.com/MicroscPSF/MicroscPSF-Py
"""
import cmath
import math
import numpy
import scipy
import scipy.integrate
import scipy.interpolate
import scipy.special

# Additions, torch friendly versions of main functions
import torch
import torch.nn.functional as F


# Internal constants.
num_basis = 100  # Number of rescaled Bessels that approximate the phase function.
rho_samples = 1000  # Number of pupil sample along the radial direction.


def differentiable_arange(start, end, step, device):
    num_elements = int((end - start) / step)
    return start + torch.arange(num_elements, device=device) * step


def calcRv(dxy, xy_size, sampling=2):
    """
    Calculate rv vector, this is 2x up-sampled.
    """
    rv_max = math.sqrt(0.5 * xy_size * xy_size) + 1
    return numpy.arange(0, rv_max * dxy, dxy / sampling)


def calcRv_torch(dxy, xy_size, sampling=2, device=torch.device("cpu")):
    """
    Calculate rv vector, this is 2x up-sampled.
    """
    rv_max = math.sqrt(0.5 * xy_size * xy_size) + 1

    return differentiable_arange(0, rv_max * dxy, dxy / sampling, device=device)
    # return torch.arange(0, rv_max * dxy, dxy / sampling, device=device)


def configure(mp, wvl):
    # Scaling factors for the Fourier-Bessel series expansion
    min_wavelength = 0.436  # microns
    scaling_factor = (
        mp["NA"] * (3 * numpy.arange(1, num_basis + 1) - 2) * min_wavelength / wvl
    )

    # Not sure this is completely correct for the case where the axial
    # location of the flourophore is 0.0.
    #
    max_rho = (
        min([mp["NA"], mp["ng0"], mp["ng"], mp["ni0"], mp["ni"], mp["ns"]]) / mp["NA"]
    )

    return [scaling_factor, max_rho]


def configure_torch(mp, wvl, device=torch.device("cpu")):
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

    mp - The microscope parameters dictionary.
    zd - Actual camera position in microns.
    """
    a = mp["NA"] * mp["zd0"] / mp["M"]  # Aperture radius at the back focal plane.
    return a * a * (mp["zd0"] - zd) / (2.0 * mp["zd0"] * zd)


def gLXYZCameraScan(mp, dxy, xy_size, zd, normalize=True, pz=0.0, wvl=0.6, zv=0.0):
    """
    NOTE: Does not work!

    Calculate 3D G-L PSF. This is models the PSF you would measure by scanning the
    camera position (changing the microscope tube length).

    This will return a numpy array with of size (zv.size, xy_size, xy_size). Note that z
    is the zeroth dimension of the PSF.

    mp - The microscope parameters dictionary.
    dxy - Step size in the XY plane.
    xy_size - Number of pixels in X/Y.
    zd - A numpy array containing the camera positions in microns.

    normalize - Normalize the PSF to unit height.
    pz - Particle z position above the coverslip (positive values only).
    wvl - Light wavelength in microns.
    zv - The (relative) z offset value of the coverslip (negative is closer to the objective).
    """
    # Calculate rv vector, this is 2x up-sampled.
    rv = calcRv(dxy, xy_size)

    # Calculate radial/Z PSF.
    PSF_rz = gLZRCameraScan(mp, rv, zd, normalize=normalize, pz=pz, wvl=wvl, zv=zv)

    # Create XYZ PSF by interpolation.
    return psfRZToPSFXYZ(dxy, xy_size, rv, PSF_rz)


def gLXYZFocalScan(mp, dxy, xy_size, zv, normalize=True, pz=0.0, wvl=0.6, zd=None):
    """
    Calculate 3D G-L PSF. This is models the PSF you would measure by scanning the microscopes
    focus.

    This will return a numpy array with of size (zv.size, xy_size, xy_size). Note that z
    is the zeroth dimension of the PSF.

    mp - The microscope parameters dictionary.
    dxy - Step size in the XY plane.
    xy_size - Number of pixels in X/Y.
    zv - A numpy array containing the (relative) z offset values of the coverslip (negative is closer to the objective).

    normalize - Normalize the PSF to unit height.
    pz - Particle z position above the coverslip (positive values only).
    wvl - Light wavelength in microns.
    zd - Actual camera position in microns. If not specified the microscope tube length is used.
    """
    # Calculate rv vector, this is 2x up-sampled.
    rv = calcRv(dxy, xy_size)

    # Calculate radial/Z PSF.
    PSF_rz = gLZRFocalScan(mp, rv, zv, normalize=normalize, pz=pz, wvl=wvl, zd=zd)

    # Create XYZ PSF by interpolation.
    return psfRZToPSFXYZ(dxy, xy_size, rv, PSF_rz)


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

    This will return a numpy array with of size (zv.size, xy_size, xy_size). Note that z
    is the zeroth dimension of the PSF.

    mp - The microscope parameters dictionary.
    dxy - Step size in the XY plane.
    xy_size - Number of pixels in X/Y.
    zv - A numpy array containing the (relative) z offset values of the coverslip (negative is closer to the objective).

    normalize - Normalize the PSF to unit height.
    pz - Particle z position above the coverslip (positive values only).
    wvl - Light wavelength in microns.
    zd - Actual camera position in microns. If not specified the microscope tube length is used.
    """
    # Calculate rv vector, this is 2x up-sampled.
    rv = calcRv_torch(dxy, xy_size, device=device)

    # Calculate radial/Z PSF.
    PSF_rz = gLZRFocalScan_torch(
        mp, rv, zv, normalize=normalize, pz=pz, wvl=wvl, zd=zd, device=device
    )

    # Create XYZ PSF by interpolation.
    return psfRZToPSFXYZ_torch(dxy, xy_size, rv, PSF_rz, device=device)


def gLXYZParticleScan(mp, dxy, xy_size, pz, normalize=True, wvl=0.6, zd=None, zv=0.0):
    """
    Calculate 3D G-L PSF. This is models the PSF you would measure by scanning a particle
    through the microscopes focus.

    This will return a numpy array with of size (zv.size, xy_size, xy_size). Note that z
    is the zeroth dimension of the PSF.

    mp - The microscope parameters dictionary.
    dxy - Step size in the XY plane.
    xy_size - Number of pixels in X/Y.
    pz - A numpy array containing the particle z position above the coverslip (positive values only)
         in microns.

    normalize - Normalize the PSF to unit height.
    wvl - Light wavelength in microns.
    zd - Actual camera position in microns. If not specified the microscope tube length is used.
    zv - The (relative) z offset value of the coverslip (negative is closer to the objective).
    """
    # Calculate rv vector, this is 2x up-sampled.
    rv = calcRv(dxy, xy_size)

    # Calculate radial/Z PSF.
    PSF_rz = gLZRParticleScan(mp, rv, pz, normalize=normalize, wvl=wvl, zd=zd, zv=zv)

    # Create XYZ PSF by interpolation.
    return psfRZToPSFXYZ(dxy, xy_size, rv, PSF_rz)


def gLZRScan(mp, pz, rv, zd, zv, normalize=True, wvl=0.6):
    """
    Calculate radial G-L at specified radius. This function is primarily designed
    for internal use. Note that only one pz, zd and zv should be a numpy array
    with more than one element. You can simulate scanning the focus, the particle
    or the camera but not 2 or 3 of these values at the same time.

    mp - The microscope parameters dictionary.
    pz - A numpy array containing the particle z position above the coverslip (positive values only).
    rv - A numpy array containing the radius values.
    zd - A numpy array containing the actual camera position in microns.
    zv - A numpy array containing the relative z offset value of the coverslip (negative is closer to the objective).


    normalize - Normalize the PSF to unit height.
    wvl - Light wavelength in microns.
    """
    [scaling_factor, max_rho] = configure(mp, wvl)
    rho = numpy.linspace(0.0, max_rho, rho_samples)

    a = (
        mp["NA"] * mp["zd0"] / math.sqrt(mp["M"] * mp["M"] + mp["NA"] * mp["NA"])
    )  # Aperture radius at the back focal plane.
    k = 2.0 * numpy.pi / wvl

    ti = zv.reshape(-1, 1) + mp["ti0"]
    pz = pz.reshape(-1, 1)
    zd = zd.reshape(-1, 1)

    opdt = OPD(mp, rho, ti, pz, wvl, zd)

    # Sample the phase
    # phase = numpy.cos(opdt) + 1j * numpy.sin(opdt)
    phase = numpy.exp(1j * opdt)

    # Define the basis of Bessel functions
    # Shape is (number of basis functions by number of rho samples)
    J = scipy.special.jv(0, scaling_factor.reshape(-1, 1) * rho)

    # Compute the approximation to the sampled pupil phase by finding the least squares
    # solution to the complex coefficients of the Fourier-Bessel expansion.
    # Shape of C is (number of basis functions by number of z samples).
    # Note the matrix transposes to get the dimensions correct.
    C, residuals, _, _ = numpy.linalg.lstsq(J.T, phase.T)

    rv = rv * mp["M"]
    b = k * a * rv.reshape(-1, 1) / zd

    # Convenience functions for J0 and J1 Bessel functions
    J0 = lambda x: scipy.special.jv(0, x)
    J1 = lambda x: scipy.special.jv(1, x)

    # See equation 5 in Li, Xue, and Blu
    denom = scaling_factor * scaling_factor - b * b
    R = (
        scaling_factor * J1(scaling_factor * max_rho) * J0(b * max_rho) * max_rho
        - b * J0(scaling_factor * max_rho) * J1(b * max_rho) * max_rho
    )
    R /= denom

    # The transpose places the axial direction along the first dimension of the array, i.e. rows
    # This is only for convenience.
    PSF_rz = (numpy.abs(R.dot(C)) ** 2).T

    # Normalize to the maximum value
    if normalize:
        PSF_rz /= numpy.max(PSF_rz)

    return PSF_rz


def gLZRScan_torch(
    mp, pz, rv, zd, zv, normalize=True, wvl=0.6, device=torch.device("cpu")
):
    """
    Calculate radial G-L at specified radius. This function is primarily designed
    for internal use. Note that only one pz, zd and zv should be a numpy array
    with more than one element. You can simulate scanning the focus, the particle
    or the camera but not 2 or 3 of these values at the same time.

    mp - The microscope parameters dictionary.
    pz - A numpy array containing the particle z position above the coverslip (positive values only).
    rv - A numpy array containing the radius values.
    zd - A numpy array containing the actual camera position in microns.
    zv - A numpy array containing the relative z offset value of the coverslip (negative is closer to the objective).

    normalize - Normalize the PSF to unit height.
    wvl - Light wavelength in microns.
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


def gLZRCameraScan(mp, rv, zd, normalize=True, pz=0.0, wvl=0.6, zv=0.0):
    """
    NOTE: Does not work!

    Calculate radial G-L at specified radius and z values. This is models the PSF
    you would measure by scanning the camera position (changing the microscope
    tube length).

    mp - The microscope parameters dictionary.
    rv - A numpy array containing the radius values.
    zd - A numpy array containing the camera positions in microns.

    normalize - Normalize the PSF to unit height.
    pz - Particle z position above the coverslip (positive values only).
    wvl - Light wavelength in microns.
    zv - The (relative) z offset value of the coverslip (negative is closer to the objective).
    """
    pz = numpy.array([pz])
    zv = numpy.array([zv])

    return gLZRScan(mp, pz, rv, zd, zv, normalize=normalize, wvl=wvl)


def gLZRFocalScan(mp, rv, zv, normalize=True, pz=0.0, wvl=0.6, zd=None):
    """
    Calculate radial G-L at specified radius and z values. This is models the PSF
    you would measure by scanning the microscopes focus.

    mp - The microscope parameters dictionary.
    rv - A numpy array containing the radius values.
    zv - A numpy array containing the (relative) z offset values of the coverslip (negative is
         closer to the objective) in microns.

    normalize - Normalize the PSF to unit height.
    pz - Particle z position above the coverslip (positive values only).
    wvl - Light wavelength in microns.
    zd - Actual camera position in microns. If not specified the microscope tube length is used.
    """
    if zd is None:
        zd = mp["zd0"]

    pz = numpy.array([pz])
    zd = numpy.array([zd])

    return gLZRScan(mp, pz, rv, zd, zv, normalize=normalize, wvl=wvl)


def gLZRFocalScan_torch(
    mp, rv, zv, normalize=True, pz=0.0, wvl=0.6, zd=None, device=torch.device("cpu")
):
    """
    Calculate radial G-L at specified radius and z values. This is models the PSF
    you would measure by scanning the microscopes focus.

    mp - The microscope parameters dictionary.
    rv - A numpy array containing the radius values.
    zv - A numpy array containing the (relative) z offset values of the coverslip (negative is
         closer to the objective) in microns.

    normalize - Normalize the PSF to unit height.
    pz - Particle z position above the coverslip (positive values only).
    wvl - Light wavelength in microns.
    zd - Actual camera position in microns. If not specified the microscope tube length is used.
    """
    if zd is None:
        zd = mp["zd0"]

    pz = torch.tensor([pz], device=device)
    zd = torch.tensor([zd], device=device)

    return gLZRScan_torch(
        mp, pz, rv, zd, zv, normalize=normalize, wvl=wvl, device=device
    )


def gLZRParticleScan(mp, rv, pz, normalize=True, wvl=0.6, zd=None, zv=0.0):
    """
    Calculate radial G-L at specified radius and z values. This is models the PSF
    you would measure by scanning the particle relative to the microscopes focus.

    mp - The microscope parameters dictionary.
    rv - A numpy array containing the radius values.
    pz - A numpy array containing the particle z position above the coverslip (positive values only)
         in microns.

    normalize - Normalize the PSF to unit height.
    wvl - Light wavelength in microns.
    zd - Actual camera position in microns. If not specified the microscope tube length is used.
    zv - The (relative) z offset value of the coverslip (negative is closer to the objective).
    """
    if zd is None:
        zd = mp["zd0"]

    zd = numpy.array([zd])
    zv = numpy.array([zv])

    return gLZRScan(mp, pz, rv, zd, zv, normalize=normalize, wvl=wvl)


def OPD(mp, rho, ti, pz, wvl, zd):
    """
    Calculate phase aberration term.

    mp - The microscope parameters dictionary.
    rho - Rho term.
    ti - Coverslip z offset in microns.
    pz - Particle z position above the coverslip in microns.
    wvl - Light wavelength in microns.
    zd - Actual camera position in microns.
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
    k = 2.0 * numpy.pi / wvl  # Wave number of emitted light.

    OPDs = pz * numpy.sqrt(ns * ns - NA * NA * rho * rho)  # OPD in the sample.
    OPDi = ti * numpy.sqrt(ni * ni - NA * NA * rho * rho) - ti0 * numpy.sqrt(
        ni0 * ni0 - NA * NA * rho * rho
    )  # OPD in the immersion medium.
    OPDg = tg * numpy.sqrt(ng * ng - NA * NA * rho * rho) - tg0 * numpy.sqrt(
        ng0 * ng0 - NA * NA * rho * rho
    )  # OPD in the coverslip.
    OPDt = a * a * (zd0 - zd) * rho * rho / (2.0 * zd0 * zd)  # OPD in camera position.

    return k * (OPDs + OPDi + OPDg + OPDt)


def OPD_torch(mp, rho, ti, pz, wvl, zd, device=torch.device("cpu")):
    """
    Calculate phase aberration term.

    mp - The microscope parameters dictionary.
    rho - Rho term.
    ti - Coverslip z offset in microns.
    pz - Particle z position above the coverslip in microns.
    wvl - Light wavelength in microns.
    zd - Actual camera position in microns.
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


def psfRZToPSFXYZ(dxy, xy_size, rv, PSF_rz):
    """
    Use interpolation to create a 3D XYZ PSF from a 2D ZR PSF.
    """
    # Create XY grid of radius values.
    c_xy = float(xy_size) * 0.5
    xy = numpy.mgrid[0:xy_size, 0:xy_size] + 0.5
    r_pixel = dxy * numpy.sqrt(
        (xy[1] - c_xy) * (xy[1] - c_xy) + (xy[0] - c_xy) * (xy[0] - c_xy)
    )

    # Create XYZ PSF by interpolation.
    PSF_xyz = numpy.zeros((PSF_rz.shape[0], xy_size, xy_size))
    for i in range(PSF_rz.shape[0]):
        psf_rz_interp = scipy.interpolate.interp1d(rv, PSF_rz[i, :])
        PSF_xyz[i, :, :] = psf_rz_interp(r_pixel.ravel()).reshape(xy_size, xy_size)

    return PSF_xyz


def psfRZToPSFXYZ_torch(dxy, xy_size, rv, PSF_rz, device=torch.device("cpu")):
    """
    Use interpolation to create a 3D XYZ PSF from a 2D ZR PSF.
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


def slowGL(mp, max_rho, rv, zv, pz, wvl, zd):
    """
    Calculate a single point in the G-L PSF using integration. This
    is primarily provided for testing / reference purposes. As the
    function name implies, this is going to be slow.

    mp - The microscope parameters dictionary.
    max_rho - The maximum rho value.
    rv - A radius value in microns.
    zv - A z offset value (of the coverslip) in microns.
    pz - Particle z position above the coverslip in microns.
    wvl - Light wavelength in microns.
    zd - Actual camera position in microns.
    """
    a = (
        mp["NA"] * mp["zd0"] / math.sqrt(mp["M"] * mp["M"] + mp["NA"] * mp["NA"])
    )  # Aperture radius at the back focal plane.
    k = 2.0 * numpy.pi / wvl
    ti = zv + mp["ti0"]

    rv = rv * mp["M"]

    def integral_fn_imag(rho):
        t1 = k * a * rho * rv / zd
        t2 = scipy.special.jv(0, t1)
        t3 = t2 * cmath.exp(1j * OPD(mp, rho, ti, pz, wvl, zd)) * rho
        return t3.imag

    def integral_fn_real(rho):
        t1 = k * a * rho * rv / zd
        t2 = scipy.special.jv(0, t1)
        t3 = t2 * cmath.exp(1j * OPD(mp, rho, ti, pz, wvl, zd)) * rho
        return t3.real

    int_i = scipy.integrate.quad(lambda x: integral_fn_imag(x), 0.0, max_rho)[0]
    int_r = scipy.integrate.quad(lambda x: integral_fn_real(x), 0.0, max_rho)[0]

    t1 = k * a * a / (zd * zd)
    return t1 * (int_r * int_r + int_i * int_i)


def gLZRFocalScanSlow(mp, rv, zv, normalize=True, pz=0.0, wvl=0.6, zd=None):
    """
    This is the integration version of gLZRFocalScan.

    mp - The microscope parameters dictionary.
    rv - A numpy array containing the radius values.
    zv - A numpy array containing the (relative) z offset values of the coverslip (negative is closer to the objective).

    normalize - Normalize the PSF to unit height.
    pz - Particle z position above the coverslip (positive values only).
    wvl - Light wavelength in microns.
    zd - Actual camera position in microns. If not specified the microscope tube length is used.
    """
    if zd is None:
        zd = mp["zd0"]

    [scaling_factor, max_rho] = configure(mp, wvl)
    rho = numpy.linspace(0.0, max_rho, rho_samples)

    psf_rz = numpy.zeros((zv.size, rv.size))
    for i in range(zv.size):
        for j in range(rv.size):
            psf_rz[i, j] = slowGL(mp, max_rho, rv[j], zv[i], pz, wvl, zd)

    if normalize:
        psf_rz = psf_rz / numpy.max(psf_rz)

    return psf_rz


def gLZRParticleScanSlow(mp, rv, pz, normalize=True, wvl=0.6, zd=None, zv=0.0):
    """
    This is the integration version of gLZRParticleScan.

    mp - The microscope parameters dictionary.
    rv - A numpy array containing the radius values.
    pz - A numpy array containing the particle z position above the coverslip (positive values only)
         in microns.

    normalize - Normalize the PSF to unit height.
    wvl - Light wavelength in microns.
    zd - Actual camera position in microns. If not specified the microscope tube length is used.
    zv - The (relative) z offset value of the coverslip (negative is closer to the objective).
    """
    if zd is None:
        zd = mp["zd0"]

    [scaling_factor, max_rho] = configure(mp, wvl)
    rho = numpy.linspace(0.0, max_rho, rho_samples)

    psf_rz = numpy.zeros((pz.size, rv.size))
    for i in range(pz.size):
        for j in range(rv.size):
            psf_rz[i, j] = slowGL(mp, max_rho, rv[j], zv, pz[i], wvl, zd)

    if normalize:
        psf_rz = psf_rz / numpy.max(psf_rz)

    return psf_rz
