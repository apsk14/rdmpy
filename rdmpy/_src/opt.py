"""
Implements all optimization functions. Primarily used by calibrate.py and deblur.py
"""

import pathlib
import gc

import numpy as np
import torch
import torch.fft as fft
from torch.utils.checkpoint import checkpoint
import matplotlib as mpl
import matplotlib.pyplot as plt

from tqdm import tqdm

from kornia.filters.sobel import spatial_gradient

from . import psf_model, util
from .. import blur

mpl.rcParams["figure.dpi"] = 500

dirname = str(pathlib.Path(__file__).parent.absolute())


def estimate_coeffs(
    calib_image,
    psf_list,
    sys_params,
    fit_params,
    higher_order=False,
    show_psfs=False,
    verbose=True,
    device=torch.device("cpu"),
):
    """
    Estimate the Seidel coefficients of the optical system given a calibration image of
    randomly scattered PSFs and their locations.

    Parameters
    ----------
    calib_image : torch.Tensor
        Calibration image of randomly scattered PSFs.

    psf_list : list of tuples
        List of PSF locations in the calibration image. Expecting xy coordinates,
        not rowcol coordinates.

    sys_params : dict
        Dictionary of optical system parameters.

    fit_params : dict
        Dictionary of fitting parameters.

    higher_order : bool
        Whether to fit higher order coefficients. The default is False.

    show_psfs : bool
        Whether to show the estimated PSFs. The default is False.

    verbose : bool
        Whether to print out progress. The default is True.

    device : torch.device
        Device to run the calibration on. The default is torch.device("cpu").

    Returns
    -------
    final_coeffs : torch.Tensor (or list if fit_params['get_inter_seidels'] is True)
        Seidel coefficients of the optical system.

    """

    if fit_params["seidel_init"] is not None:
        coeffs = torch.tensor(fit_params["seidel_init"], device=device)
    else:
        if fit_params["init"] == "zeros":
            coeffs = torch.zeros((fit_params["num_seidel"], 1), device=device)
        elif fit_params["init"] == "random":
            coeffs = torch.rand((fit_params["num_seidel"], 1), device=device)
        else:
            raise NotImplementedError
    if higher_order:
        higher_order_coeffs = torch.zeros((8, 1), device=device)
        higher_order_coeffs.requires_grad = True
    else:
        higher_order_coeffs = None

    coeffs.requires_grad = True

    if higher_order:
        optimizer = torch.optim.Adam([coeffs, higher_order_coeffs], lr=fit_params["lr"])
    else:
        optimizer = torch.optim.Adam([coeffs], lr=fit_params["lr"])
    l2_loss_fn = torch.nn.MSELoss()
    l1_loss_fn = torch.nn.L1Loss()

    if fit_params["plot_loss"]:
        losses = []

    if fit_params["get_inter_seidels"]:
        inter_seidels = []

    iterations = (
        tqdm(range(fit_params["iters"])) if verbose else range(fit_params["iters"])
    )

    for iter in iterations:
        # forward pass
        if coeffs.shape[0] < 6:
            psfs_estimate = psf_model.compute_rdm_psfs(
                torch.cat((coeffs, torch.zeros(6 - coeffs.shape[0], 1, device=device))),
                desired_list=psf_list,
                stack=False,
                sys_params=sys_params,
                higher_order=higher_order_coeffs,
                device=coeffs.device,
            )
        else:
            psfs_estimate = psf_model.compute_rdm_psfs(
                coeffs,
                desired_list=psf_list,
                stack=False,
                sys_params=sys_params,
                higher_order=higher_order_coeffs,
                device=coeffs.device,
            )
        # loss
        loss = l1_loss_fn(
            util.normalize(sum(psfs_estimate).float()), util.normalize(calib_image)
        ) + fit_params["reg"] * l2_loss_fn(coeffs, -coeffs)

        if fit_params["plot_loss"]:
            losses += [loss.detach().cpu()]

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if fit_params["get_inter_seidels"]:
            if coeffs.shape[0] < 6:
                inter_seidels += [
                    torch.cat(
                        (coeffs, torch.zeros(6 - coeffs.shape[0], 1, device=device))
                    ).detach()
                ]
            else:
                inter_seidels += [coeffs.detach().cpu().detach()]

    if show_psfs:
        psf_est = sum(psfs_estimate) / (len(psfs_estimate))
        psf_est_mask = psf_est.detach().cpu().clone()
        psf_est_mask[psf_est_mask > np.quantile(psf_est_mask, 0.99)] = 1
        psfs_gt = calib_image.detach().cpu() * psf_est_mask

        plt.subplot(1, 2, 2)
        plt.tight_layout()
        plt.axis("off")
        plt.imshow(psf_est.detach().cpu(), cmap="inferno")
        plt.gca().set_title("Seidel PSFs")

        plt.subplot(1, 2, 1)
        plt.tight_layout()
        plt.axis("off")
        plt.imshow(psfs_gt, cmap="inferno")
        plt.gca().set_title("Measured PSFs")
        plt.show()

    if fit_params["plot_loss"]:
        plt.figure()
        plt.plot(range(len(losses)), losses)
        plt.show()
    del psfs_estimate
    del psfs_gt

    if fit_params["get_inter_seidels"]:
        final_coeffs = inter_seidels
    else:
        if coeffs.shape[0] < 6:
            final_coeffs = torch.cat(
                (coeffs, torch.zeros(6 - coeffs.shape[0], 1, device=device))
            ).detach()
        else:
            final_coeffs = coeffs.detach()

    if higher_order:
        return final_coeffs, higher_order_coeffs.detach()
    return final_coeffs


def fit_gl_model(
    psf_stack,
    psf_locs,
    psf_xy_dim,
    gl_params,
    gl_opt=[],
    iters=100,
    lr=5e-2,
    verbose=True,
    device=torch.device("cpu"),
):
    """
    Fit the parameters of the light-sheet Gibson-lanni PSF model to a stack of PSFs.

    Parameters
    ----------
    psf_stack : torch.Tensor
        Stack of PSFs to fit the model to.

    psf_locs : list of tuples
        List of PSF locations in the calibration image. Expecting xy coordinates,
        not rowcol coordinates.

    psf_xy_dim : int
        Size of the PSF in the xy dimensions.

    gl_params : dict
        Dictionary of parameters for the Gibson-Lanni PSF model.

    gl_opt : list of str
        List of which gl_params to optimize.

    iters : int
        Number of iterations to run the optimization for.

    lr : float
        Learning rate for the optimization.

    verbose : bool
        Whether to print out progress. The default is True.

    device : torch.device
        Device to run the optimization on. The default is torch.device("cpu").

    Returns
    -------
    waist : torch.Tensor
        Waist of the PSF.

    spread : torch.Tensor
        Spread of the PSF.

    gl_params : dict
        Dictionary of parameters for the Gibson-Lanni PSF model.
    """

    torch.autograd.set_detect_anomaly(True)

    l2_loss_fn = torch.nn.MSELoss()

    best_waist = 0.1
    best_spread = 1.0

    waist = torch.tensor([best_waist], device=device)
    spread = torch.tensor([best_spread], device=device)

    if ((6 * waist) / (gl_params["dz"] * 1e-3)) < 1:
        waist.data = (gl_params["dz"] * 1e-3) / 6

    waist.requires_grad = True
    spread.requires_grad = True

    for param in gl_opt:
        gl_params[param].requires_grad = True

    optimizer = torch.optim.Adam([waist, spread] + list(gl_params.values()), lr=lr)

    iterations = tqdm(range(iters)) if verbose else range(iters)

    for iter in iterations:
        # produce candidate PSFs
        loss = 0

        for loc in psf_locs:
            lateral_pos = (loc[0] - psf_stack.shape[0] // 2) * gl_params["dx"]

            candidate_psf = psf_model.compute_GL_psf(
                lateral_pos=lateral_pos,
                waist=waist,
                spread=spread,
                gl_params=gl_params,
                xy_dim=psf_xy_dim,
                zmax=psf_stack.shape[2],
                oversample=2,
                device=device,
            )

            # compare candidate with gt psf by cropping out the appropriate section of the gt psf, ensuring edges are handled correctly
            buffer = 4
            # concentare buffer zeros in every dimensions
            candidate_psf = torch.nn.functional.pad(
                candidate_psf,
                (buffer, buffer),
                mode="constant",
                value=0,
            )

            psf_dim_z = candidate_psf.shape[2]
            if loc[0] < psf_xy_dim // 2:
                left_crop = psf_xy_dim // 2 - loc[0]
                candidate_psf = candidate_psf[left_crop:, :, :]
            if loc[0] > psf_stack.shape[0] - psf_xy_dim // 2:
                right_crop = loc[0] - psf_stack.shape[0] + psf_xy_dim // 2
                candidate_psf = candidate_psf[:-right_crop, :, :]
            if loc[1] < psf_xy_dim // 2:
                left_crop = psf_xy_dim // 2 - loc[1]
                candidate_psf = candidate_psf[:, left_crop:, :]
            if loc[1] > psf_stack.shape[1] - psf_xy_dim // 2:
                right_crop = loc[1] - psf_stack.shape[1] + psf_xy_dim // 2
                candidate_psf = candidate_psf[:, :-right_crop, :]

            if loc[2] < psf_dim_z // 2:
                left_crop = psf_dim_z // 2 - loc[2]
                candidate_psf = candidate_psf[:, :, left_crop:]
            if loc[2] > psf_stack.shape[2] - psf_dim_z / 2:
                right_crop = (
                    loc[2] - psf_stack.shape[2] + psf_dim_z // 2 + int(psf_dim_z % 2)
                )
                candidate_psf = candidate_psf[:, :, :-right_crop]

            comparison_psf = psf_stack[
                (loc[0] - psf_xy_dim // 2)
                * int(loc[0] >= psf_xy_dim // 2) : (
                    (loc[0] + psf_xy_dim // 2)
                    if loc[0] <= psf_stack.shape[0] - psf_xy_dim // 2
                    else None
                ),
                (loc[1] - psf_xy_dim // 2)
                * int(loc[1] >= psf_xy_dim // 2) : (
                    (loc[1] + psf_xy_dim // 2)
                    if loc[1] <= psf_stack.shape[1] - psf_xy_dim // 2
                    else None
                ),
                (loc[2] - psf_dim_z // 2)
                * int(loc[2] >= psf_dim_z // 2) : (
                    (loc[2] + psf_dim_z // 2 + int(psf_dim_z % 2))
                    if loc[2] < psf_stack.shape[2] - psf_dim_z // 2
                    else None
                ),
            ]

            loss += l2_loss_fn(
                candidate_psf / candidate_psf.max(),
                comparison_psf / comparison_psf.max(),
            )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if ((6 * waist.data) / (gl_params["dz"])) < 1:
            waist.data[0] = (gl_params["dz"]) / 6

        if spread.data < 1e-6:
            spread.data[0] = 1e-6

    if verbose:
        print("waist: ", waist.item())
        print("spread: ", spread.item())
    for key in gl_opt:
        value = gl_params[key]
        if verbose:
            print(key, ": ", value.item())
        gl_params[key] = value.detach()

    return (
        waist.detach(),
        spread.detach(),
        gl_params,
    )


def volume_recon(
    measurement,
    psf_stack,
    opt_params,
    warm_start=None,
    verbose=True,
    device=torch.device("cpu"),
):
    """
    Deblur a volume using either deconvolution or ring deconvolution

    Parameters
    ----------
    measurement : torch.Tensor
        Measurement to be deblurred.

    psf_data : torch.Tensor
        Stack of 3D PSFs to be used for deblurring.

    opt_params : dict
        Dictionary of optimization parameters.

    warm_start : torch.Tensor
        Warm start for the optimization. The default is None.

    verbose : bool
        Whether to print out progress. The default is True.

    device : torch.device
        Device to run the reconstruction on. The default is torch.device("cpu").

    Returns
    -------
    estimate : torch.Tensor
        Deblurred volume.
    """

    dim = measurement.shape

    if warm_start is not None:
        estimate = warm_start.clone()
    else:
        if opt_params["init"] == "measurement":
            estimate = measurement.clone()
        elif opt_params["init"] == "zero":
            estimate = torch.zeros(dim, device=device)
        elif opt_params["init"] == "noise":
            estimate = torch.randn(dim, device=device)
        else:
            raise NotImplementedError

    estimate.requires_grad = True

    if opt_params["optimizer"] == "adam":
        optimizer = torch.optim.Adam([estimate], lr=opt_params["lr"])
    elif opt_params["optimizer"] == "sgd":
        optimizer = torch.optim.SGD([estimate], lr=opt_params["lr"])
    else:
        raise NotImplementedError

    center_psf = psf_stack[-1]
    dim = estimate.shape
    diff_2 = (dim[2] - center_psf.shape[2]) // 2
    diff_1 = (dim[1] - center_psf.shape[1]) // 2
    diff_0 = (dim[0] - center_psf.shape[0]) // 2
    # take psf and pad it with zeros on either side till it has the same last dimension as the object
    center_psf = torch.nn.functional.pad(
        center_psf,
        (diff_2 + 1, diff_2, diff_1, diff_1, diff_0, diff_0),
        mode="constant",
        value=0,
    )
    loss_fn = torch.nn.MSELoss()

    losses = []

    if opt_params["plot_loss"]:
        losses = []

    iterations = (
        tqdm(range(opt_params["iters"])) if verbose else range(opt_params["iters"])
    )
    for it in iterations:

        measurement_guess = blur.sheet_convolve(estimate, psf_stack, device=device)

        loss_main = loss_fn(measurement_guess, measurement)
        loss = (
            loss_main
            + tv(estimate, opt_params["tv_reg"])
            + opt_params["l2_reg"] * torch.norm(estimate)
            + opt_params["l1_reg"] * torch.sum(torch.abs(estimate))
        )

        if opt_params["plot_loss"]:
            losses += [loss.detach().cpu()]

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        with torch.no_grad():
            # project onto [0,1]
            estimate.data[estimate.data < 0] = 0
            # upper proejction
            if opt_params["upper_projection"]:
                estimate.data[estimate.data > 1] = 1

    if opt_params["plot_loss"]:
        plt.figure()
        plt.plot(range(len(losses)), losses)
        plt.show()

    final = util.normalize(estimate.detach().cpu().float())

    del estimate
    gc.collect()

    return final


def image_recon(
    measurement,
    psf_data,
    model,
    opt_params,
    warm_start=None,
    verbose=True,
    device=torch.device("cpu"),
):
    """
    Deblur an image using either deconvolution or ring deconvolution.

    Parameters
    ----------
    measurement : torch.Tensor
        Measurement to be deblurred. Should be (N,N).

    psf_data : torch.Tensor
        Stack of rotationatal Fourier transforms of the PSFs if model is 'lri',
        otherwise single center PSF if model is 'lsi'.

    model : str
        Either 'lri' or 'lsi'.

    opt_params : dict
        Dictionary of optimization parameters.

    warm_start : torch.Tensor
        Warm start for the optimization. The default is None.

    verbose : bool
        Whether to print out progress. The default is True.

    device : torch.device
        Device to run the reconstruction on. The default is torch.device("cpu").

    Returns
    -------
    estimate : torch.Tensor
        Deblurred image. Will be (N,N).
    """

    dim = measurement.shape
    torch.autograd.set_detect_anomaly(True)

    if warm_start is not None:
        estimate = warm_start.clone()
    else:
        if opt_params["init"] == "measurement":
            estimate = measurement.clone()
        elif opt_params["init"] == "zero":
            estimate = torch.zeros(dim, device=device)
        elif opt_params["init"] == "noise":
            estimate = torch.randn(dim, device=device)
        else:
            raise NotImplementedError

    estimate.requires_grad = True

    if opt_params["optimizer"] == "adam":
        optimizer = torch.optim.Adam([estimate], lr=opt_params["lr"])
    elif opt_params["optimizer"] == "sgd":
        optimizer = torch.optim.SGD([estimate], lr=opt_params["lr"])
    else:
        raise NotImplementedError

    loss_fn = torch.nn.MSELoss()

    losses = []

    if opt_params["plot_loss"]:
        losses = []

    iterations = (
        tqdm(range(opt_params["iters"])) if verbose else range(opt_params["iters"])
    )

    crop = lambda x: (
        x[
            ...,
            opt_params["crop"] : -opt_params["crop"],
            opt_params["crop"] : -opt_params["crop"],
        ]
        if opt_params["crop"] > 0
        else x
    )

    for it in iterations:
        # forward pass and loss
        if model == "lsi":
            measurement_guess = blur.convolve(estimate, psf_data)
        elif model == "lri":
            measurement_guess = blur.ring_convolve(
                estimate,
                psf_data,
                device=device,
            )
        elif model == "lri_patch":
            measurement_guess = blur.ring_convolve_patch(
                estimate,
                psf_data,
                patch_size=opt_params["patch_size"],
                device=device,
            )
        else:
            raise NotImplementedError

        loss = (
            loss_fn(
                crop(measurement_guess),
                crop(measurement),
            )
            + tv(crop(estimate), opt_params["tv_reg"])
            + opt_params["l2_reg"] * torch.norm(crop(estimate))
            + opt_params["l1_reg"] * torch.sum(torch.abs(crop(estimate)))
        )

        if opt_params["plot_loss"]:
            losses += [loss.detach().cpu()]

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # project onto [0,1]
        estimate.data[estimate.data < 0] = 0
        # upper proejction
        if opt_params["upper_projection"]:
            estimate.data[estimate.data > 1] = 1

    if opt_params["plot_loss"]:
        plt.figure()
        plt.plot(range(len(losses)), losses)
        plt.show()

    final = util.normalize(estimate.detach().cpu().float())

    del estimate
    gc.collect()

    return final


def blind_recon(
    measurement, opt_params, sys_params, verbose=True, device=torch.device("cpu")
):
    """
    Deblur an image using a blind deconvolution approach (no calibration needed).

    Parameters
    ----------
    measurement : torch.Tensor
        Measurement to be deblurred.

    opt_params : dict
        Dictionary of optimization parameters.

    sys_params : dict
        Dictionary of optical system parameters.

    verbose : bool
        Whether to print out progress. The default is True.

    device : torch.device
        Device to run the reconstruction on. The default is torch.device("cpu").

    Returns
    -------
    recon : torch.Tensor
        Deblurred image.

    psfs_estimate : torch.Tensor
        Estimated PSF.
    """

    if opt_params["seidel_init"] is None:
        coeffs = torch.zeros((1, 1), device=device)

    coeffs.requires_grad = True

    optimizer = torch.optim.Adam([coeffs], lr=opt_params["lr"])

    if opt_params["plot_loss"]:
        losses = []

    iterations = (
        tqdm(range(opt_params["iters"])) if verbose else range(opt_params["iters"])
    )

    for iter in iterations:
        # forward pass
        psfs_estimate = psf_model.compute_rdm_psfs(
            torch.cat((coeffs, torch.zeros(5, 1, device=device))),
            desired_list=[(0, 0)],
            sys_params=sys_params,
            device=coeffs.device,
        )[0]
        recon = wiener(
            ((measurement - 0.5) * 2), psfs_estimate, balance=opt_params["balance"]
        )
        recon = (recon / 2) + 0.5  # back-scale

        # loss (maximizing acutance)
        loss = -torch.mean(
            torch.abs(spatial_gradient(recon[None, None, :, :], mode="diff"))
        )

        if opt_params["plot_loss"]:
            losses += [loss.detach().cpu()]

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if opt_params["plot_loss"]:
        plt.figure()
        plt.plot(range(len(losses)), losses)
        plt.show()

    if verbose:
        print("Fitted seidel coefficient: " + str(coeffs[0].detach().cpu()))

    return (
        recon.detach().cpu(),
        psfs_estimate.detach().cpu(),
    )


def fit_gaussian(
    calib_image,
    sys_params,
    fit_params,
    show_psfs=False,
    verbose=True,
    device=torch.device("cpu"),
):
    """
    Estimate the Seidel coefficients of the optical system given a calibration image of
    randomly scattered PSFs and their locations.

    Parameters
    ----------
    calib_image : torch.Tensor
        Calibration image of randomly scattered PSFs.

    sys_params : dict
        Dictionary of optical system parameters.

    fit_params : dict
        Dictionary of fitting parameters.

    show_psfs : bool
        Whether to show the estimated PSFs. The default is False.

    verbose : bool
        Whether to print out progress. The default is True.

    device : torch.device
        Device to run the calibration on. The default is torch.device("cpu").

    Returns
    -------
    var : torch.Tensor
        Estimated variance of the Gaussian PSF.

    final_psf : np.ndarray
        Estimated PSF.

    """

    var = torch.tensor([1e-2], device=device)
    var.requires_grad = True

    optimizer = torch.optim.Adam([var], lr=fit_params["lr"])
    l2_loss_fn = torch.nn.MSELoss()

    if fit_params["plot_loss"]:
        losses = []

    iterations = (
        tqdm(range(fit_params["iters"])) if verbose else range(fit_params["iters"])
    )

    for iter in iterations:
        var_2 = var * torch.ones([2, 1], device=device)
        psf_estimate = psf_model.make_gaussian(
            var_2, calib_image.shape[0], sys_params["L"], device=device
        )
        loss = l2_loss_fn(
            ((psf_estimate / psf_estimate.max()).float()),
            (calib_image / calib_image.max()).float(),
        ) + fit_params["reg"] * l2_loss_fn(var, -var)

        if fit_params["plot_loss"]:
            losses += [loss.detach().cpu()]

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    final_psf = psf_estimate.detach().cpu()

    if show_psfs:

        plt.subplot(1, 2, 2)
        plt.tight_layout()
        plt.axis("off")
        plt.imshow(final_psf, cmap="inferno")
        plt.gca().set_title("Seidel PSFs")

        plt.subplot(1, 2, 1)
        plt.tight_layout()
        plt.axis("off")
        plt.imshow(calib_image.detach().cpu().numpy(), cmap="inferno")
        plt.gca().set_title("Measured PSFs")
        plt.show()

    if fit_params["plot_loss"]:
        plt.figure()
        plt.plot(range(len(losses)), losses)
        plt.show()

    del psf_estimate
    del calib_image

    return var, final_psf


def wiener(image, psf, balance=5e-4, reg=None, is_real=True, clip=True):
    """
    Apply a wiener filter to an image using the psf.

    Parameters
    ----------
    image : torch.Tensor
        Image to be deblurred. Should be (M,N).

    psf : torch.Tensor
        PSF to be used for deblurring. Should be (M,N).

    balance : float
        Balance parameter for the wiener filter. The default is 5e-4.

    reg : torch.Tensor
        Regularization parameter for the wiener filter. The default is None.

    is_real : bool
        Whether the PSF is real or complex. The default is True.

    clip : bool
        Whether to clip the output to [-1,1]. The default is True.

    Returns
    -------
    deconv : torch.Tensor
        Deblurred image.

    Notes
    -----
    This function is adapted from the `wiener` function in the `restoration` module of
    scikit-image. https://github.com/scikit-image/scikit-image/blob/v0.21.0/skimage/restoration/deconvolution.py


    """

    if reg is None:
        reg, _ = laplacian(image.ndim, image.shape, is_real=is_real)
        reg = reg.to(device=image.device)

    trans_func = ir2tf(psf, image.shape, is_real=is_real).to(device=image.device)

    wiener_filter = torch.conj(trans_func) / (
        torch.abs(trans_func) ** 2 + balance * torch.abs(reg) ** 2
    )

    deconv = fft.irfftn(wiener_filter * fft.rfftn(image))

    if clip:
        deconv[deconv > 1] = 1
        deconv[deconv < -1] = -1

    return deconv


### HELPER FUNCTIONS ###


def tv(img, weight):
    """
    Computes the total variation of an image.

    Parameters
    ----------
    img : torch.Tensor
        Image to compute the total variation of.

    weight : float
        Weighting factor for the total variation.

    Returns
    -------
    tv : torch.Tensor
        Total variation of the image.
    """

    tv_h = ((img[1:, :] - img[:-1, :]).abs()).sum()
    tv_w = ((img[:, 1:] - img[:, :-1]).abs()).sum()
    return weight * (tv_h + tv_w)


def laplacian(ndim, shape, is_real=True):
    """
    Compute the Laplacian operator in the Fourier domain.

    Parameters
    ----------
    ndim : int
        Number of dimensions of the image.

    shape : tuple
        Shape of the image.

    is_real : bool
        Whether the PSF is real or complex. The default is True.

    Returns
    -------
    laplacian : torch.Tensor
        Laplacian operator in the Fourier domain.

    impr : torch.Tensor
        Laplacian operator in the spatial domain.

    Notes
    -----
    This function is taken from the `laplacian` function in the `restoration` module of
    scikit-image. https://github.com/scikit-image/scikit-image/blob/v0.21.0/skimage/restoration/deconvolution.py
    """

    impr = torch.zeros([3] * ndim)
    for dim in range(ndim):
        idx = tuple(
            [slice(1, 2)] * dim + [slice(None)] + [slice(1, 2)] * (ndim - dim - 1)
        )
        impr[idx] = torch.tensor([-1.0, 0.0, -1.0]).reshape(
            [-1 if i == dim else 1 for i in range(ndim)]
        )
    impr[(slice(1, 2),) * ndim] = 2.0 * ndim
    return ir2tf(impr, shape, is_real=is_real), impr


def ir2tf(imp_resp, shape, dim=None, is_real=True):
    """
    Implement the ir2tf function from the `restoration` module of scikit-image.

    Parameters
    ----------
    imp_resp : torch.Tensor
        Impulse response of the system.

    shape : tuple
        Shape of the image.

    dim : int
        Last axis to compute the FFT over. The default is None.

    is_real : bool
        Whether the PSF is real or complex. The default is True.

    Returns
    -------
    tf : torch.Tensor
        Transfer function of the system.

    Notes
    -----
    This function is taken from the `ir2tf` function in the `restoration` module of
    scikit-image. https://github.com/scikit-image/scikit-image/blob/v0.21.0/skimage/restoration/uft.py

    """

    if not dim:
        dim = imp_resp.ndim
    # Zero padding and fill
    if imp_resp.shape != shape:
        irpadded = torch.zeros(shape)
        irpadded[tuple([slice(0, s) for s in imp_resp.shape])] = imp_resp
    else:
        irpadded = imp_resp
    # Roll for zero convention of the fft to avoid the phase
    # problem. Work with odd and even size.
    for axis, axis_size in enumerate(imp_resp.shape):
        if axis >= imp_resp.ndim - dim:
            irpadded = torch.roll(
                irpadded, shifts=-int(np.floor(axis_size / 2)), dims=axis
            )

    return fft.rfftn(irpadded, dim=list(range(-dim, 0)))
