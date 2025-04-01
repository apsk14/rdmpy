"""Implementation of Seidel calibration"""

import numpy as np
import torch
import torch.fft as fft
import matplotlib as mpl
from tqdm import tqdm
from skimage.feature import corner_peaks

from ._src import opt, seidel, util
import gc

mpl.rcParams["figure.dpi"] = 500


def calibrate(
    calib_image,
    dim,
    model="lri",
    get_psf_data=True,
    num_seidel=4,
    fit_params={},
    sys_params={},
    show_psfs=False,
    downsample=1,
    verbose=True,
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
        Either 'lsi', 'lri', or 'gaussian' for the type of PSF model to use.
        'lsi'/'gaussian' returns a single PSF at the center of the image, while 'lri' returns a
        radial line of PSF RoFTs. Use 'lri' for ring deconvolution, and 'lsi' or 'gaussian' for
        standard deconvolution.

    get_psf_data : bool, optional
        Whether to return the PSFs or just the seidel coefficients.

    num_seidel : int, optional
        Number of Seidel coefficients to fit. Default is 4 (excludes distortion and defocus by default)

    fit_params : dict, optional
        Parameters for the seidel fitting procedure. See `opt.py` for details.

    sys_params : dict, optional
        Parameters for the optical system. See `seidel.py` for details.

    show_psfs : bool, optional
        Whether to show the PSFs estimated by the Seidel fit.

    downsample : int, optional
        Factor by which to downsample the PSFs after fitting. Useful for saving memory.

    verbose : bool, optional
        Whether to print out progress.

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
        "L": 0,
        "lamb": 0.55e-6,
        "NA": 0.5,
    }
    radius_over_z = np.tan(np.arcsin(def_sys_params["NA"]))
    def_sys_params["L"] = ((dim) * (def_sys_params["lamb"])) / (4 * (radius_over_z))

    def_sys_params.update(sys_params)

    # parameters which are used for the seidel fitting procedure
    def_fit_params = {
        "sys_center": [
            calib_image.shape[0] // 2,
            calib_image.shape[1] // 2,
        ],  # center of the system
        "centered_psf": False,  # whether there is a PSF that is centered in the calibration image
        "min_distance": 30,  # minimum distance between fitted PSFs
        "threshold": 0.2,  # threshold for detecting PSFs
        "disk": 2,  # disk size for detecting PSFs (make smaller if PSFs are very small (only a few pixels))
        "num_seidel": num_seidel,  # number of seidel coefficients to fit, up to 6
        "init": "zeros",  # initialization scheme for the seidel coefficients
        "seidel_init": None,  # direct initialization for the seidel coefficients
        "iters": 300,  # number of iterations for the fitting procedure
        "lr": 1e-2,  # learning rate for the fitting procedure
        "reg": 0,  # regularization for the fitting procedure
        "plot_loss": False,  # whether to plot the loss during the fitting procedure
        "get_inter_seidels": False,  # whether to return intermediate seidel coefficients
    }
    def_fit_params.update(fit_params)

    # seperating out individual PSFs from the calibration image
    psf_locations, calib_image = util.get_calib_info(calib_image, dim, def_fit_params)

    if model == "gaussian":
        # get psf location closest to the center
        center = (dim // 2, dim // 2)
        center_psf = calib_image.copy()
        # set to zero everything outside a center crop of size buffer
        buffer = 10
        center_psf[0 : center[0] - buffer, :] = 0
        center_psf[center[0] + buffer :, :] = 0
        center_psf[:, 0 : center[1] - buffer] = 0
        center_psf[:, center[1] + buffer :] = 0
        center_psf = center_psf / np.sum(center_psf)
        # fit the gaussian PSF
        var, center_psf = opt.fit_gaussian(
            center_psf,
            def_sys_params,
            def_fit_params,
            show_psfs=True,
            verbose=True,
            device=device,
        )
        if downsample < 1:
            center_psf = seidel.make_gaussian(
                [var, var], int(dim // downsample), def_sys_params["L"], device=device
            )

        return center_psf / center_psf.sum()

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
        print("Fitted seidel coefficients: " + str(seidel_coeffs.abs().detach().cpu()))

    if get_psf_data:
        psf_data = get_psfs(
            seidel_coeffs,
            dim,
            model,
            sys_params=def_sys_params,
            downsample=downsample,
            verbose=verbose,
            device=device,
        )
        return seidel_coeffs.abs(), psf_data

    else:
        if get_inter_seidels:
            seidel_coeffs = coeffs

    return seidel_coeffs.abs()


def calibrate_sdm(
    psf_stack,
    psf_dim,
    model="gl",
    get_psfs=True,
    iters=100,
    NA=0.2,
    wavelength=550e-6,
    device=torch.device("cpu"),
):

    psf_stack[psf_stack < np.quantile(psf_stack, 0.9)] = 0

    psf_locs = get_psf_centers(psf_stack, min_distance=20)
    # order psf_locs by 0th index
    psf_locs = psf_locs[psf_locs[:, 0].argsort()]
    # get psf_locs cloest to the center
    center = torch.tensor(
        [psf_stack.shape[0] // 2, psf_stack.shape[1] // 2, psf_stack.shape[2] // 2],
        device=device,
    )
    psf_locs = torch.tensor(psf_locs, device=device)
    min_idx = torch.argmin(torch.norm(psf_locs.float() - center.float(), dim=1))

    print("min_idx:", min_idx)
    psf_center = psf_locs[min_idx].unsqueeze(0)[0]

    # crop psf_stack centered on psf_center of size psf_dim, psf_dim, z_max
    center_psf = psf_stack[
        psf_center[0] - psf_dim // 2 : psf_center[0] + psf_dim // 2,
        psf_center[1] - psf_dim // 2 : psf_center[1] + psf_dim // 2,
        psf_center[2] - psf_dim // 2 : psf_center[2] + psf_dim // 2,
    ]

    if model == "deconvolution":
        return center_psf

    if model == "interpolation":
        if not get_psfs:
            return psf_locs
        psf_stack = torch.tensor(psf_stack, device=device).float()
        psf_cube = []
        # loop from center to top and set psf to an interpolation of the two nearest psfs
        for i in range(psf_stack.shape[0]):
            curr_y = i + psf_stack.shape[0] // 2
            closest_index = torch.argmin(torch.abs(psf_locs[:, 0] - curr_y))
            if psf_locs[closest_index][0] < curr_y:
                weight_1 = torch.abs(curr_y - psf_locs[closest_index][0])
                weight_2 = torch.abs(curr_y - psf_locs[closest_index - 1][0])
                weight_1 = weight_1 / (weight_1 + weight_2)
                weight_2 = weight_2 / (weight_1 + weight_2)
                curr_psf = weight_2 * isolate_psf(
                    psf_stack, psf_locs[closest_index], psf_dim
                ) + weight_1 * isolate_psf(
                    psf_stack, psf_locs[closest_index - 1], psf_dim
                )
                curr_psf[curr_psf < torch.quantile(curr_psf, 0.9)] = 0

            elif (
                psf_locs[closest_index][0] > curr_y
                and closest_index < len(psf_locs) - 1
            ):
                weight_1 = torch.abs(curr_y - psf_locs[closest_index][0])
                weight_2 = torch.abs(curr_y - psf_locs[closest_index + 1][0])
                weight_1 = weight_1 / (weight_1 + weight_2)
                weight_2 = weight_2 / (weight_1 + weight_2)
                curr_psf = weight_2 * isolate_psf(
                    psf_stack, psf_locs[closest_index], psf_dim
                ) + weight_1 * isolate_psf(
                    psf_stack, psf_locs[closest_index + 1], psf_dim
                )
                curr_psf[curr_psf < torch.quantile(curr_psf, 0.9)] = 0
            else:
                curr_psf = isolate_psf(psf_stack, psf_locs[closest_index], psf_dim)
                curr_psf[curr_psf < torch.quantile(curr_psf, 0.9)] = 0

            psf_cube += [curr_psf]

        return psf_cube, psf_locs

    waist, spread, defocus_rate, coeffs, gl_params, dx, dz = opt.fit_beam(
        psf_stack,
        psf_locs,
        psf_dim,
        coeffs=None,
        NA=NA,
        wavelength=wavelength,
        dz=None,
        iters=iters,
        lr=2e-4,
        verbose=True,
        device=device,
        grid_search=False,
    )

    if get_psfs:
        psf_cube = get_ls_psfs(
            coeffs,
            waist,
            spread,
            defocus_rate,
            dx,
            dz,
            psf_dim,
            psf_stack.shape[2],
            psf_stack.shape[0],
            wavelength,
            NA,
            get_center_vals=False,
            gl_params=gl_params,
            device=device,
        )

        return psf_cube, psf_locs

    return coeffs, waist, spread, defocus_rate, psf_locs, gl_params, dx, dz


def get_ls_psfs(
    coeffs,
    waist,
    spread,
    defocus_rate,
    dx,
    dz,
    dim,
    zmax,
    num_psfs,
    wavelength=550e-6,
    NA=0.2,
    get_center_vals=False,
    gl_params=None,
    device=torch.device("cpu"),
):
    # check if coeffs is tensor, if not make it one
    if not torch.is_tensor(coeffs):
        coeffs = torch.tensor(coeffs, device=device)

    # get params for the imaging system
    radius_over_z = torch.tan(
        torch.arcsin(torch.tensor(NA))
    )  # radius length in the fourier domain
    L = ((dim) * (wavelength)) / (4 * (radius_over_z))
    # dx = L / dim
    # dz = dx * 2

    psf_cube_list = []
    if get_center_vals:
        psf_centers_values = []

    for i in tqdm(range(num_psfs)):
        norm_x = i / num_psfs
        x = i * (L / dim)
        curr_psf = seidel.get_ls_psfs(
            coeffs,
            waist,
            spread,
            torch.tensor([defocus_rate], device=device),
            x,
            norm_x,
            dx,
            dz,
            dim,
            zmax,
            wavelength,
            NA,
            gl_params,
            device,
        )
        psf_cube_list += [curr_psf]
        if get_center_vals:
            psf_centers_values += [torch.sum(curr_psf * curr_psf)]
    if get_center_vals:
        return psf_cube_list, torch.tensor(psf_centers_values, device=device)

    return psf_cube_list


def get_psfs(
    seidel_coeffs,
    dim,
    model,
    sys_params={},
    downsample=1,
    psf_data=None,
    verbose=True,
    higher_order=None,
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

    downsample : int, optional
        Factor by which to downsample the PSFs after fitting. Useful for saving memory.

    psf_data : torch.Tensor, optional
        Precomputed PSFs. If not provided, will be computed.

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
        "L": 0,
        "lamb": 0.55e-6,
        "NA": 0.5,
    }
    radius_over_z = np.tan(np.arcsin(def_sys_params["NA"]))
    def_sys_params["L"] = ((dim) * (def_sys_params["lamb"])) / (4 * (radius_over_z))
    def_sys_params.update(sys_params)

    if not torch.is_tensor(seidel_coeffs):
        seidel_coeffs = torch.tensor(seidel_coeffs).to(device)

    if model == "lsi":
        point_list = [(0, 0)]  # just the center PSF
    elif model == "lri":
        rs = np.linspace(
            0, (dim / 2), int(dim // abs(downsample)), endpoint=False, retstep=False
        )
        point_list = [(r, -r) for r in rs]  # radial line of PSFs
    else:
        raise (NotImplementedError)

    if verbose:
        print("rendering PSFs...")

    if model == "lsi":
        buffer = 0
    elif model == "lri":
        buffer = 2

    if psf_data is None:
        psf_data = seidel.compute_psfs(
            seidel_coeffs,
            point_list,
            sys_params=def_sys_params,
            polar=(model == "lri"),
            stack=True,
            buffer=buffer,
            downsample=downsample,
            higher_order=higher_order,
            verbose=verbose,
            device=device,
        )

    # prep the PSFs for outputing to the user
    if model == "lsi":
        psf_data = psf_data[0].to(device)
    if model == "lri":
        # here compute the RoFT of each PSF in-place (torch.rfft is memory inefficient)
        for i in range(psf_data.shape[0]):
            temp_rft = fft.rfft(psf_data[i, 0:-2, :], dim=0)
            psf_data[i, 0 : psf_data.shape[1] // 2, :] = torch.real(temp_rft)
            psf_data[i, psf_data.shape[1] // 2 :, :] = torch.imag(temp_rft)

        del temp_rft
        gc.collect()

    return psf_data


def isolate_psf(psf_stack, psf_loc, psf_dim):
    return psf_stack[
        psf_loc[0] - psf_dim // 2 : psf_loc[0] + psf_dim // 2,
        psf_loc[1] - psf_dim // 2 : psf_loc[1] + psf_dim // 2,
        psf_loc[2] - psf_dim // 2 : psf_loc[2] + psf_dim // 2,
    ]


def get_psf_centers(psf_stack, min_distance=2, threshold=0.1):
    psf_locs = corner_peaks(
        psf_stack,
        min_distance=min_distance,
        indices=True,
        threshold_rel=threshold,
    )
    return psf_locs
