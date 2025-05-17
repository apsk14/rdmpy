"""Implementation of PSF calibration"""

import numpy as np
import torch
import torch.fft as fft
import matplotlib as mpl
from skimage.feature import corner_peaks
from skimage.morphology import erosion, disk
import torch.nn.functional as F

import gc
from tqdm import tqdm

from ._src import opt, psf_model, util, polar_transform
import pdb

import matplotlib.pyplot as plt

mpl.rcParams["figure.dpi"] = 300


def calibrate_rdm(
    calib_img,
    dim,
    model="lri",
    sys_center=None,
    get_psfs=True,
    num_seidel=4,
    patch_size=0,
    higher_order=False,
    fit_params={},
    sys_params={},
    show_psfs=False,
    downsample=1,
    verbose=True,
    device=torch.device("cpu"),
):
    """Fit image of randomly scattered point sources to Seidel coefficients and return PSF data if specified.

    This function takes a calibration image of randomly scattered point sources and fits it to a Seidel aberration coefficients.
    It returns the Seidel coefficients and optionally the center PSF (if using the LSI or gaussian model) or PSF ROFTs (if using the LRI model).
    The PSF ROFTs can be plugged directly into ring deconvolution. If the coefficients are already known use `rdmpy.calibrate.get_rdm_psfs`
    to get the PSF data. If PSFs at every radius are already known, they can be converted to ROFTs using `rdmpy.calibrate.psf_to_ROFT`.

    Parameters
    ----------
    calib_img : np.ndarray
        Calibration image, ideally an image of sparse, randomly-placed point sources. Can be any size (M, N) but will be cropped
        to (dim, dim) and centered to the center of the image by default or sys_center if specified.

    dim : int
        Desired sidelength of each PSF image. Note that it enforces square images.

    model : str
        Either 'lsi', 'lri', or 'gaussian' for the type of PSF model to use. Using 'lsi' or 'gaussian'
        returns a single PSF at the center of the image, while 'lri' returns a stack PSF RoFTs.
        Use 'lri' with ring deconvolution, and 'lsi' or 'gaussian' with standard deconvolution.

    sys_center : tuple
        Center of the optical system. If not specified, the center of the calibration image will be used.

    get_psfs : bool
        Whether to return the PSFs or just the seidel coefficients. Should be False if PSFs are not needed in order
        to save time and memory.

    num_seidel : int
        Number of Seidel coefficients to fit. Default is 4 (excludes distortion and defocus by default as they are less common)

    patch_size : int
        Size of the isoplanatic annuli. If 0, no patching will be done. If > 0, the PSFs will be computed in patches.

    fit_params : dict
        Parameters for the seidel fitting procedure.

    sys_params : dict
        Parameters for the optical system.

    show_psfs : bool
        Whether to show the PSFs estimated by the Seidel fit.


    downsample : int
        Factor by which to downsample the PSFs after fitting. Useful for fitting to higher resolution PSFs but then using
        lower resolution PSFs for deconvolution.

    verbose : bool
        Whether to print out progress.

    device : torch.device
        Device to run the calibration on.

    Returns
    -------
    seidel_coeffs : torch.Tensor
        Seidel coefficients of the optical system. Will be (6,1).

    psf_data : torch.Tensor
        Only returns if `get_psfs` is True. PSFs of the optical system. If `model` is 'lsi',
        this is a single PSF (dim, dim). If `model` is 'lri', this is a stack of PSF RoFTs (dim, 4*dim, dim).

    Notes
    -----
    PSF generation using the LRI model can be slow. Use a GPU if possible, especially for larger images, i.e., greaer than (256,256).
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

    if sys_center is not None:
        fit_params["sys_center"] = sys_center

    # parameters which are used for the seidel fitting procedure
    def_fit_params = {
        "sys_center": [
            calib_img.shape[0] // 2,
            calib_img.shape[1] // 2,
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
    psf_locations, calib_img = get_calib_info(calib_img, dim, def_fit_params)

    if model == "gaussian":
        # get psf location closest to the center
        center = (dim // 2, dim // 2)
        center_psf = calib_img.copy()
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
            center_psf = psf_model.make_gaussian(
                [var, var], int(dim // downsample), def_sys_params["L"], device=device
            )

        return center_psf / center_psf.sum()

    # seidel fitting

    if verbose:
        print("fitting seidel coefficients...")
    coeffs = opt.estimate_coeffs(
        torch.tensor(calib_img, device=device),
        psf_list=psf_locations,
        sys_params=def_sys_params,
        fit_params=def_fit_params,
        higher_order=higher_order,
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

    if get_psfs:
        psf_data = get_rdm_psfs(
            seidel_coeffs,
            dim,
            model,
            patch_size=patch_size,
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


def get_rdm_psfs(
    seidel_coeffs,
    dim,
    model,
    patch_size=0,
    sys_params={},
    downsample=1,
    higher_order=None,
    verbose=True,
    device=torch.device("cpu"),
):
    """Return PSF ROFTs for a given set of Seidel coefficients.

    This function takes a set of Seidel coefficients and returns the PSF ROFTs for the optical system. There is one
    PSF ROFT for each radius in the image. The PSF ROFTs can be used directly in ring deconvolution.

    Parameters
    ----------
    seidel_coeffs : torch.Tensor
        Seidel coefficients of the optical system. Should be (6,1) with coefficients: sphere, coma, astigmatism, field curvature, distortion, defocus.

    dim : int
        Desired sidelength of each PSF image. Note that it enforces square images.

    model : str
        Either 'lsi' or 'lri' for the type of PSF model to use. LSI model will return a single PSF at the center of the image,
        while LRI model will return a stack of PSF RoFTs.

    patch_size : int
        Size of the isoplanatic annuli. If 0, no patching will be done. If > 0, the PSFs will be computed in patches.

    sys_params : dict
        Parameters for the optical system.

    downsample : int
        Factor by which to downsample the PSFs after fitting. Useful for saving memory.

    higher_order : bool
        Whether to include higher order aberrations in the PSF computation. If True, will include an additional 8 coefficients.

    verbose : bool
        Whether to print out progress.

    device : torch.device

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

    patch_based = patch_size > 0 and patch_size <= dim // abs(downsample)

    if not torch.is_tensor(seidel_coeffs):
        seidel_coeffs = torch.tensor(seidel_coeffs).to(device)

    if model == "lsi":
        point_list = [(0, 0)]  # just the center PSF
    elif model == "lri":
        rs = np.linspace(
            0, (dim / 2), int(dim // abs(downsample)), endpoint=False, retstep=False
        )

        if patch_based:
            rs = rs[::patch_size]

        point_list = [(r, -r) for r in rs]  # radial line of PSFs
    else:
        raise (NotImplementedError)

    if verbose:
        print("rendering PSFs...")

    if model == "lri":
        buffer = 2
    else:
        buffer = 0

    psf_data = psf_model.compute_rdm_psfs(
        seidel_coeffs,
        point_list,
        sys_params=def_sys_params,
        polar=(model == "lri" and not patch_based),
        stack=True,
        buffer=buffer,
        shift=not patch_based,
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

        if patch_based:
            pad = dim // 2
            psf_data = F.pad(psf_data, (0, pad, 0, pad))
            for i in range(psf_data.shape[0]):
                temp_rft = fft.rfft2(psf_data[i, :, 0:-buffer])
                psf_data[i, :, 0 : psf_data.shape[-1] // 2] = torch.real(temp_rft)
                psf_data[i, :, psf_data.shape[-1] // 2 :] = torch.imag(temp_rft)
        else:
            for i in range(psf_data.shape[0]):
                temp_rft = fft.rfft(psf_data[i, 0:-2, :], dim=0)
                psf_data[i, 0 : psf_data.shape[1] // 2, :] = torch.real(temp_rft)
                psf_data[i, psf_data.shape[1] // 2 :, :] = torch.imag(temp_rft)

            del temp_rft
        gc.collect()

    return psf_data


def get_calib_info(calib_image, dim, fit_params):
    """
    Localizes PSFs in calibration image.

    Parameters
    ----------
    calib_image : np.ndarray
        Calibration image.

    dim : tuple
        Dimension of desired PSFs. Should be square (N,N).

    fit_params : dict
        Dictionary of parameters for calibration.

    Returns
    -------
    coord_list : list
        List of coordinates of PSFs in calibration image (xy) format.

    calib_image : np.ndarray
        Cropped calibration image.

    """

    psf = calib_image.copy()
    psf[psf < 0] = 0
    psf[psf < np.quantile(psf, 0.9)] = 0
    if fit_params["disk"] <= 0:
        raw_coord = corner_peaks(
            psf,
            min_distance=fit_params["min_distance"],
            indices=True,
            threshold_rel=fit_params["threshold"],
        )
    else:
        raw_coord = corner_peaks(
            erosion(psf, disk(fit_params["disk"])),
            min_distance=fit_params["min_distance"],
            indices=True,
            threshold_rel=fit_params["threshold"],
        )
    distances = np.sqrt(np.sum(np.square(raw_coord - fit_params["sys_center"]), axis=1))
    if fit_params["centered_psf"]:
        center = raw_coord[np.argmin(distances), :]
        print(center)
    else:
        center = fit_params["sys_center"].copy()
    if dim // 2 > center[0]:
        PAD = dim // 2 - center[0]
        calib_image = np.pad(calib_image, ((PAD, 0), (0, 0)))
        center[0] += PAD
    if dim // 2 > center[1]:
        PAD = dim // 2 - center[1]
        calib_image = np.pad(calib_image, ((0, 0), (PAD, 0)))
        center[1] += PAD
    if dim // 2 + center[0] > calib_image.shape[0]:
        PAD = dim // 2 + center[0] - calib_image.shape[0]
        calib_image = np.pad(calib_image, ((0, PAD), (0, 0)))
        center[0] -= PAD
    if dim // 2 + center[1] > calib_image.shape[1]:
        PAD = dim // 2 + center[1] - calib_image.shape[1]
        calib_image = np.pad(calib_image, ((0, 0), (0, PAD)))
        center[1] -= PAD

    calib_image = calib_image[
        center[0] - dim // 2 : center[0] + dim // 2,
        center[1] - dim // 2 : center[1] + dim // 2,
    ]

    coord_list = []
    for i in range(raw_coord.shape[0]):
        if (
            np.abs(raw_coord[i, 1] - center[1]) < dim // 2
            and np.abs(center[0] - raw_coord[i, 0]) < dim // 2
        ):
            coord_list += [(raw_coord[i, 1] - center[1], center[0] - raw_coord[i, 0])]

    calib_image[calib_image < 0] = 0
    calib_image[calib_image < np.quantile(calib_image, 0.9)] = 0
    calib_image = (calib_image / calib_image.sum()) * len(coord_list)

    return coord_list, calib_image


def isolate_psf(psf_stack, psf_loc, psf_dim):
    """Isolate a PSF at a particular location from a volume containing multiple PSFs.

    Parameters
    ----------
    psf_stack : torch.Tensor
        Volume containing multiple PSFs.

    psf_loc : tuple
        Location of the center of the desired PSF in the stack.

    psf_dim : int
        Size of the PSF to isolate. Note this is a single number because the PSF is assumed to be in a cube.

    Returns
    -------
    psf : torch.Tensor
        Isolated PSF from the stack.
    """

    return psf_stack[
        psf_loc[0] - psf_dim // 2 : psf_loc[0] + psf_dim // 2,
        psf_loc[1] - psf_dim // 2 : psf_loc[1] + psf_dim // 2,
        psf_loc[2] - psf_dim // 2 : psf_loc[2] + psf_dim // 2,
    ]


def get_psf_centers(psf_stack, min_distance=2, threshold=0.1):
    """Get the locations of the PSFs in a stack.

    This function uses the skimage.feature.corner_peaks function to find the locations of the PSFs in a stack.

    Parameters
    ----------
    psf_stack : torch.Tensor
        Volme containing multiple PSFs.

    min_distance : int
        Minimum distance between PSFs. Default is 2.

    threshold : float
        Threshold for detecting PSFs. Default is 0.1.

    Returns
    -------
    psf_locs : torch.Tensor
        Locations of the PSFs in the stack. Shape is (num_psfs, 3).
    """

    psf_locs = corner_peaks(
        psf_stack,
        min_distance=min_distance,
        indices=True,
        threshold_rel=threshold,
    )

    return psf_locs


def calibrate_sdm(
    psf_stack,
    psf_xy_dim,
    model="gl",
    get_psfs=True,
    gl_params={},
    gl_opt=[],
    iters=50,
    lr=1e-2,
    oversample=2,
    verbose=True,
    device=torch.device("cpu"),
):
    """Use a calibration volume of randomly scattered point sources to render PSFs at all points along the light-sheet.

    This function takes a volume of randomly scattered PSFs and and either interpolates or fits a light-sheet Gibson-Lanni model
    in order to generate a stack of PSFs at all points along the light-sheet. The resulting PSFs can be used directly in sheet deconvolution.

    Parameters
    ----------
    psf_stack : torch.Tensor
        Volume containing multiple PSFs. The light-sheet varying axis should be the 0th axis of the stack.
        Moreover, it should be centered on the light-sheet waist.


    psf_xy_dim : int
        Lateral size of the PSF to fit. Note this is a single number because the PSF is assumed to be in square.
        The axial size will be set to the last dimension of the PSF stack.

    model : str
        Either 'deconvolution' to just get the center PSF, 'gl' to fit the light-sheet Gibson-Lanni model or
        'interpolation' to linearly interpolate PSFs from the stack.

    get_psfs : bool
        Whether to return the PSFs or just the parameters of the model. Should be False if PSFs are not needed in order
        to save time and memory.

    gl_params : dict
        Parameters for the Gibson-Lanni model. See https://github.com/MicroscPSF/MicroscPSF-Py for details.
        The parameters are:
        M : float
            Magnification of the system.
        NA : float
            Numerical aperture of the system.
        ng0 : float
            Refractive index of the coverslip (design value).
        ng : float
            Refractive index of the coverslip (experimental value).
        ni0 : float
            Refractive index of the immersion medium (design value).
        ni : float
            Refractive index of the immersion medium (experimental value).
        ns : float
            Refractive index of the specimen.
        ti0 : float
            Working distance (immersion medium thickness) (design value).
        tg : float
            Coverslip thickness (experimental value).
        tg0 : float
            Coverslip thickness (design value).
        zd0 : float
            Tube length (design value).
        dx : float
            Pixel pitch in the lateral direction.
        dz : float
            Pixel pitch in the axial direction.
        wavelength : float
            Wavelength of the light used in the system.

    gl_opt : list
        Which, if any, of the gl parameters to optimize. Default is [] which means none of the parameters will be optimized.
        Caution: certain parameters may cause NAN values if optimized for a particular system.

    iters : int
        Number of iterations for the optimization.

    lr : float
        Learning rate for the optimization.

    oversample : int
        Oversampling factor for the PSF data. Allows for more accurate PSF generation at the cost of speed.

    verbose : bool
        Whether to print out progress.

    device : torch.device
        Device to run the calibration on. Default is CPU.

    Returns
    -------
    center_psf : torch.Tensor
        Center PSF of the optical system. Only returned if `model` is 'deconvolution'.

    psf_cube : torch.Tensor
        List of 3D PSFs along the light-sheet axis, from center to top. If `model` is 'gl', this is a stack of PSFs at different radii.
        Only returned if `get_psfs` is True.

    psf_locs : torch.Tensor
        Locations of the PSFs in the stack. Shape is (num_psfs, 3). Only returned if `get_psfs` is True.

    waist : torch.Tensor
        Waist parameter of the light-sheet. Only returned if `model` is 'gl'.

    spread : torch.Tensor
        Spread parameter of the light-sheet. Only returned if `model` is 'gl'.

    gl_params : dict
        Updated parameters for the Gibson-Lanni model.

    Notes
    -----
    Note that the fitting can be imperfect if the Gibson-Lanni parameters are not set correctly. Please
    choose them according to the system you are using and if known use their theoretical values.
    """

    # clean up the psf_stack
    psf_stack[psf_stack < np.quantile(psf_stack, 0.9)] = 0

    # find the locations of the PSFs in the stack
    psf_locs = get_psf_centers(psf_stack, min_distance=20)

    # order psf_locs by 0th index
    psf_locs = psf_locs[psf_locs[:, 0].argsort()]

    # get psf_locs cloest to the center to extraxt the center PSF
    center = torch.tensor(
        [psf_stack.shape[0] // 2, psf_stack.shape[1] // 2, psf_stack.shape[2] // 2],
        device=device,
    )
    psf_locs = torch.tensor(psf_locs, device=device)
    min_idx = torch.argmin(torch.norm(psf_locs.float() - center.float(), dim=1))
    psf_center = psf_locs[min_idx].unsqueeze(0)[0]

    # get center_psf
    center_psf = psf_stack[
        psf_center[0] - psf_xy_dim // 2 : psf_center[0] + psf_xy_dim // 2,
        psf_center[1] - psf_xy_dim // 2 : psf_center[1] + psf_xy_dim // 2,
        psf_center[2] - psf_xy_dim // 2 : psf_center[2] + psf_xy_dim // 2,
    ]

    if model == "deconvolution":
        if verbose:
            print("Returning center PSF...")
        return center_psf

    if model == "interpolation":
        if verbose:
            print("Interpolating PSFs...")
        psf_stack = torch.tensor(psf_stack, device=device).float()
        psf_cube = []
        # loop from center to top and set psf to an interpolation of the two nearest psfs
        for i in range(psf_stack.shape[0] // 2):
            curr_y = i + psf_stack.shape[0] // 2
            closest_index = torch.argmin(torch.abs(psf_locs[:, 0] - curr_y))
            if psf_locs[closest_index][0] < curr_y:
                weight_1 = torch.abs(curr_y - psf_locs[closest_index][0])
                weight_2 = torch.abs(curr_y - psf_locs[closest_index - 1][0])
                weight_1 = weight_1 / (weight_1 + weight_2)
                weight_2 = weight_2 / (weight_1 + weight_2)
                curr_psf = weight_2 * isolate_psf(
                    psf_stack, psf_locs[closest_index], psf_xy_dim
                ) + weight_1 * isolate_psf(
                    psf_stack, psf_locs[closest_index - 1], psf_xy_dim
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
                    psf_stack, psf_locs[closest_index], psf_xy_dim
                ) + weight_1 * isolate_psf(
                    psf_stack, psf_locs[closest_index + 1], psf_xy_dim
                )
                curr_psf[curr_psf < torch.quantile(curr_psf, 0.9)] = 0
            else:
                curr_psf = isolate_psf(psf_stack, psf_locs[closest_index], psf_xy_dim)
                curr_psf[curr_psf < torch.quantile(curr_psf, 0.9)] = 0

            psf_cube += [curr_psf]
        # norm_factor = torch.sum(psf_cube[0])
        # psf_cube = [curr_psf / norm_factor for curr_psf in psf_cube]

        return psf_cube, psf_locs

    if model == "gl":
        psf_stack = torch.tensor(psf_stack, device=device).float()
        print("Fitting GL model...")
        # default parameters for the gibson-lanni model based on the light-sheet system from the paper.
        def_gl_params = {
            "M": torch.tensor(25.0, device=device),
            "NA": torch.tensor(1.1, device=device),
            "ng0": torch.tensor(1.0, device=device),  # coverslip RI design value
            "ng": torch.tensor(1.0, device=device),  # coverslip RI experimental value
            "ni0": torch.tensor(
                1.33, device=device
            ),  # immersion medium RI design value
            "ni": torch.tensor(
                1.33, device=device
            ),  # immersion medium RI experimental value
            "ns": torch.tensor(1.5, device=device),  # specimen refractive index (RI)
            "ti0": torch.tensor(
                2000.0, device=device
            ),  # microns, working distance (immersion medium thickness) design value
            "tg": torch.tensor(
                0.0, device=device
            ),  # microns, coverslip thickness experimental value
            "tg0": torch.tensor(
                0.0, device=device
            ),  # microns, coverslip thickness design value
            "zd0": torch.tensor(
                200.0 * 1.0e3, device=device
            ),  # tube length design value
            "dx": torch.tensor(
                0.104, device=device
            ),  # microns, pixel pitch in the lateral direction
            "dz": torch.tensor(
                0.3, device=device
            ),  # microns, pixel pitch in the axial direction
            "wavelength": torch.tensor(0.561, device=device),  # microns, wavelength
        }

        def_gl_params.update(gl_params)

        waist, spread, gl_params = opt.fit_gl_model(
            psf_stack,
            psf_locs,
            psf_xy_dim,
            gl_params=def_gl_params,
            gl_opt=gl_opt,
            iters=iters,
            lr=lr,
            verbose=verbose,
            device=device,
        )
    else:
        raise (NotImplementedError)

    if get_psfs:
        psf_cube = get_sdm_psfs(
            waist=waist,
            spread=spread,
            psf_xy_dim=psf_xy_dim,
            ls_dim=psf_stack.shape[0],
            z_dim=psf_stack.shape[2],
            gl_params=gl_params,
            oversample=oversample,
            device=device,
        )
        return psf_cube, psf_locs

    return waist, spread, gl_params


def get_sdm_psfs(
    waist,
    spread,
    ls_dim,
    psf_xy_dim,
    z_dim,
    gl_params={},
    oversample=2,
    verbose=True,
    device=torch.device("cpu"),
):
    r"""Generate a stack of PSFs along the light-sheet axis given light-sheet Gibson-Lanni parameters.


    Parameters
    ----------
    waist : torch.Tensor
        The waist parameter of the light-sheet.

    spread : torch.Tensor
        The spread parameter of the light-sheet.

    ls_dim : int
        This is the dimension along the light-sheet varying axis of the sample you would like to sheet deconvolve.
        This number should be even. The number of PSFs returned will be `ls_dim // 2`.

    psf_xy_dim : int
        The lateral size of the PSF to use. Note this is a single number because the PSF is assumed to fit in a square crop.
        The axial size will be set adaptively according to `spread`.

    max_z_dim : int
        The largest axial size of the PSF to use. Should be no larger than the z-dimension of the sample you would like to sheet deconvolve.

    gl_params : dict
        Parameters for the Gibson-Lanni model.

    oversample : int
        Oversampling factor for the PSF data. Allows for more accurate PSF generation at the cost of speed.

    verbose : bool
        Whether to print out progress.

    device : torch.device
        Device to run the calibration on. Default is CPU.

    Returns
    -------
    psf_cube_list : list
        List of 3D PSFs along the light-sheet axis, from center to top. This is a list of tensors of shape (psf_xy_dim, psf_xy_dim, z_dim). The PSFs are
        in the order of the light-sheet varying axis, i.e., the first PSF is at the center of the light-sheet and the last PSF is at the top of the light-sheet.
    """

    def_gl_params = {
        "M": torch.tensor(25.0, device=device),
        "NA": torch.tensor(1.1, device=device),
        "ng0": torch.tensor(1.0, device=device),  # coverslip RI design value
        "ng": torch.tensor(1.0, device=device),  # coverslip RI experimental value
        "ni0": torch.tensor(1.33, device=device),  # immersion medium RI design value
        "ni": torch.tensor(
            1.33, device=device
        ),  # immersion medium RI experimental value
        "ns": torch.tensor(1.5, device=device),  # specimen refractive index (RI)
        "ti0": torch.tensor(
            2000.0, device=device
        ),  # microns, working distance (immersion medium thickness) design value
        "tg": torch.tensor(
            0.0, device=device
        ),  # microns, coverslip thickness experimental value
        "tg0": torch.tensor(
            0.0, device=device
        ),  # microns, coverslip thickness design value
        "zd0": torch.tensor(200.0 * 1.0e3, device=device),  # tube length design value
        "dx": torch.tensor(
            0.104, device=device
        ),  # microns, pixel pitch in the lateral direction
        "dz": torch.tensor(
            0.3, device=device
        ),  # microns, pixel pitch in the axial direction
        "wavelength": torch.tensor(0.561, device=device),  # microns, wavelength
    }

    def_gl_params.update(gl_params)

    psf_cube_list = psf_model.compute_sdm_psfs(
        waist,
        spread,
        def_gl_params,
        num_psfs=ls_dim // 2,
        xy_dim=psf_xy_dim,
        zmax=z_dim,
        oversample=oversample,
        verbose=verbose,
        device=device,
    )

    return psf_cube_list
