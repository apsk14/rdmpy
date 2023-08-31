"""Implementation of Seidel calibration"""

import numpy as np
import torch
import torch.fft as fft

from ._src import opt, seidel, util


def calibrate(
    calib_image,
    dim,
    model="lri",
    get_psf_data=True,
    num_seidel=4,
    fit_params={},
    sys_params={},
    verbose=True,
    show_psfs=False,
    device=torch.device("cpu"),
):
    """

    Parameters
    ----------
    calib_image : torch.Tensor
        Calibration image, ideally an image of sparse, randomly-placed point sources.
        Can be any size (M, N) but will be cropped to (dim, dim) for the Seidel fitting.

    dim : int
        Desired sidelength of each PSF image. Note that it enforces square images.

    model : str, optional
        Either 'lsi' or 'lri' for the type of PSF model to use.
        'lsi' returns a single PSF at the center of the image, while 'lri' returns a
        radial line of PSF RoFTs. Use 'lri' for ring deconvolution, and 'lsi' for
        standard deconvolution.

    get_psf_data : bool, optional
        Whether to return the PSFs or just the seidel coefficients.

    fit_params : dict, optional
        Parameters for the seidel fitting procedure. See `opt.py` for details.

    sys_params : dict, optional
        Parameters for the optical system. See `seidel.py` for details.

    verbose : bool, optional
        Whether to print out progress.

    show_psfs : bool, optional
        Whether to show the PSFs estimated by the Seidel fit.

    device : torch.device, optional
        Device to run the calibration on.

    Returns
    -------
    seidel_coeffs : torch.Tensor
        Seidel coefficients of the optical system. Will be (6,1).

    psf_data : torch.Tensor
        PSFs of the optical system. If `model` is 'lsi', this is a single PSF.
        If `model` is 'lri', this is a stack of PSF RoFTs. Optional.


    """

    # default parameters which describe the optical system.
    def_sys_params = {
        "samples": dim,
        "L": 1e-3,
        "lamb": 0.55e-6,
        "pupil_radius": ((dim) * (0.55e-6) * (100e-3)) / (4 * (1e-3)),
        "z": 100e-3,
    }
    def_sys_params.update(sys_params)

    # parameters which are used for the seidel fitting procedure
    def_fit_params = {
        "sys_center": [calib_image.shape[0] // 2, calib_image.shape[1] // 2],
        "centered_psf": False,
        "min_distance": 30,
        "threshold": 0.2,
        "num_seidel": num_seidel,
        "init": "zeros",
        "seidel_init": None,
        "iters": 300,
        "lr": 1e-2,
        "reg": 0,
        "plot_loss": False,
        "get_inter_seidels": False,
    }
    def_fit_params.update(fit_params)

    # seperating out individual PSFs from the calibration image
    psf_locations, calib_image = util.get_calib_info(calib_image, dim, def_fit_params)

    # seidel fitting
    if verbose:
        print("fitting seidel coefficients...")
    coeffs = opt.estimate_coeffs(
        calib_image,
        psf_list=psf_locations,
        sys_params=def_sys_params,
        fit_params=def_fit_params,
        show_psfs=show_psfs,
        device=device,
    )
    get_inter_seidels = def_fit_params["get_inter_seidels"]
    if get_inter_seidels:
        seidel_coeffs = coeffs[-1]
    else:
        seidel_coeffs = coeffs
    if verbose:
        print("Fitted seidel coefficients: " + str(seidel_coeffs.detach().cpu()))

    if get_psf_data:
        psf_data = get_psfs(
            seidel_coeffs,
            dim,
            model,
            sys_params=def_sys_params,
            verbose=verbose,
            device=device,
        )

        return seidel_coeffs, psf_data

    else:
        if get_inter_seidels:
            seidel_coeffs = coeffs

        return seidel_coeffs


def get_psfs(
    seidel_coeffs,
    dim,
    model,
    sys_params={},
    verbose=False,
    device=torch.device("cpu"),
):
    """

    Parameters
    ----------
    seidel_coeffs : torch.Tensor
        Seidel coefficients of the optical system.

    dim : int
        Desired sidelength of each PSF image. Note that it enforces square images.

    model : str
        Either 'lsi' or 'lri' for the type of PSF model to use.

    sys_params : dict, optional
        Parameters for the optical system. See `seidel.py` for details.

    verbose : bool, optional

    device : torch.device, optional

    Returns
    -------
    psf_data : torch.Tensor
        PSFs of the optical system. If `model` is 'lsi', this is a single PSF.
        If `model` is 'lri', this is a stack of PSF RoFTs.

    """

    # default parameters which describe the optical system.
    def_sys_params = {
        "samples": dim,
        "L": 1e-3,
        "lamb": 0.55e-6,
        "pupil_radius": ((dim) * (0.55e-6) * (100e-3)) / (4 * (1e-3)),
        "z": 100e-3,
    }
    def_sys_params.update(sys_params)

    if model == "lsi":
        point_list = [(0, 0)]  # just the center PSF
    elif model == "lri":
        rs = np.linspace(0, (dim / 2), dim, endpoint=False, retstep=False)
        point_list = [(r, -r) for r in rs]  # radial line of PSFs
    else:
        raise (NotImplementedError)

    if verbose:
        print("rendering PSFs...")

    psf_data = seidel.compute_psfs(
        seidel_coeffs.cpu(),
        point_list,
        sys_params=def_sys_params,
        polar=(model == "lri"),
        stack=False,
        verbose=verbose,
    )

    # prep the PSFs for outputing to the user
    if model == "lsi":
        psf_data = psf_data[0].to(device)
    if model == "lri":
        psf_data = torch.stack(psf_data)
        psf_data = fft.rfft(psf_data, dim=1).to(
            device
        )  # here we store the PSF ROFT to save computation time later.

    return psf_data
