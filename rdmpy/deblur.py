"""Implementation of deblur models"""

import pathlib
import os

import torch
import numpy as np

from ._src import opt, util
from .calibrate import get_psfs
from .dl_models.DeepRD.DeepRD import UNet as DeepRD


dirname = str(pathlib.Path(__file__).parent.absolute())


def sheet_deconvolve(
    image_stack,
    psf_stack,
    tv_reg=0,
    l2_reg=0,
    l1_reg=0,
    iters=150,
    lr=1e-3,
    opt_params={},
    method="autograd",
    warm_start=None,
    verbose=True,
    center_vals=None,
    device=torch.device("cpu"),
):
    """
    Ring deconvolves an image with a stack of PSFs.

    Parameters
    ----------
    image : np.ndarray or torch.Tensor
        The image to be deconvolved. Must be (N,N), (C,N,N), or (B,C,N,N).

    psf_roft : torch.Tensor
        The stack of PSFs to deconvolve the image with. The PSFs should be in the
        Rotational Fourier domain. Should be (N, M, L) where N is the number of PSFs,
        M is the number of angles, and is L is number of radii the in the RoFT.

    tv_reg : float, optional
        The Total Variation regularization parameter. Default is 1e-10.

    l2_reg : float, optional
        The L2 norm regularization parameter. Default is 1e-10.

    l1_reg : float, optional
        The L1 norm regularization parameter. Default is 0.

    iters : int, optional
        The number of iterations to run the optimization. Default is 150.

    opt_params : dict, optional
        Ad optimization/regularization parameters to use for deconvolution.
        See `opt.py` for details.

    use_batch_conv : bool, optional
        Whether to use batched or unbatched lri convolution. The default is False.

    process : bool, optional
        Whether to process the image before deconvolution. Default is True.

    hot_pixel : bool, optional
        Whether to remove hot pixels from the image. Default is False.

    verbose : bool, optional
        Whether to display a progress bar.

    device : torch.device, optional
        The device to use for the computation.

    Returns
    -------
    recon : torch.Tensor
        The ring deconvolved image. Will be (N,N).

    Notes
    -----
    An implementation of `Ring Deconvolution Microscopy:
    An Exact Solution for Spatially-Varying Aberration Correction"
    https://arxiv.org/abs/2206.08928

    """

    # default optimization parameters
    def_opt_params = {
        "iters": iters,  # number of iterations to run the optimization
        "optimizer": "adam",  # which optimizer to use for the iterative optimization
        "lr": lr,  # learning rate of the optimizer
        "init": "measurement",  # initialization of the reconstruction before optimization
        "crop": 0,  # How much to crop out when considering the optimization loss
        "tv_reg": tv_reg,  # Total variation regularization parameter
        "l2_reg": l2_reg,  # L2 norm regularization parameter
        "l1_reg": l1_reg,  # L1 norm regularization parameter
        "plot_loss": False,  # Whether to plot the per-iteration loss during optimization
        "fraction": False,  # If true, computes fractional forward passes for lower memory consumption
        "upper_projection": False,  # If true, projects the image to [0,1] to prevent hot pixels from lowering contrast
    }
    def_opt_params.update(opt_params)

    # if process:
    #     image_stack = (
    #         util.process(image_stack, dim=image_stack.shape[-2:], hot_pix=hot_pixel)
    #         * 0.9
    #     )

    # if not torch.is_tensor(image_stack):
    #     image_stack = torch.tensor(image_stack)

    if warm_start is not None:
        warm_start = torch.tensor(warm_start)
        if warm_start.device is not device:
            warm_start = warm_start.to(device)

    if method == "autograd":
        recon = opt.volume_recon(
            image_stack,
            psf_stack,
            opt_params=def_opt_params,
            warm_start=warm_start,
            device=device,
            verbose=verbose,
        )
    elif method == "manual":
        recon = opt.volume_recon_manual(
            image_stack,
            psf_stack,
            opt_params=def_opt_params,
            warm_start=warm_start,
            device=device,
            verbose=verbose,
        )
    elif method == "cg":
        recon = opt.volume_recon_cg_normal(
            image_stack,
            psf_stack,
            opt_params=def_opt_params,
            warm_start=warm_start,
            preconditioner="jacobi",
            center_vals=center_vals,
            device=device,
            verbose=verbose,
        )
    elif method == "scipy":
        recon = opt.volume_recon_scipy(
            image_stack, psf_stack, opt_params, device=device
        )

    return recon


def ring_deconvolve(
    image,
    psf_roft,
    tv_reg=1e-10,
    l2_reg=1e-10,
    l1_reg=0,
    iters=150,
    opt_params={},
    warm_start=None,
    use_batch_conv=False,
    process=True,
    hot_pixel=False,
    verbose=True,
    device=torch.device("cpu"),
):
    """
    Ring deconvolves an image with a stack of PSFs.

    Parameters
    ----------
    image : np.ndarray or torch.Tensor
        The image to be deconvolved. Must be (N,N), (C,N,N), or (B,C,N,N).

    psf_roft : torch.Tensor
        The stack of PSFs to deconvolve the image with. The PSFs should be in the
        Rotational Fourier domain. Should be (N, M, L) where N is the number of PSFs,
        M is the number of angles, and is L is number of radii the in the RoFT.

    tv_reg : float, optional
        The Total Variation regularization parameter. Default is 1e-10.

    l2_reg : float, optional
        The L2 norm regularization parameter. Default is 1e-10.

    l1_reg : float, optional
        The L1 norm regularization parameter. Default is 0.

    iters : int, optional
        The number of iterations to run the optimization. Default is 150.

    opt_params : dict, optional
        Ad optimization/regularization parameters to use for deconvolution.
        See `opt.py` for details.

    use_batch_conv : bool, optional
        Whether to use batched or unbatched lri convolution. The default is False.

    process : bool, optional
        Whether to process the image before deconvolution. Default is True.

    hot_pixel : bool, optional
        Whether to remove hot pixels from the image. Default is False.

    verbose : bool, optional
        Whether to display a progress bar.

    device : torch.device, optional
        The device to use for the computation.

    Returns
    -------
    recon : torch.Tensor
        The ring deconvolved image. Will be (N,N).

    Notes
    -----
    An implementation of `Ring Deconvolution Microscopy:
    An Exact Solution for Spatially-Varying Aberration Correction"
    https://arxiv.org/abs/2206.08928

    """
    if len(psf_roft.shape) != 3:
        raise ValueError("Ring deconvolution needs a radial stack of PSF RoFTs")

    # default optimization parameters
    def_opt_params = {
        "iters": iters,  # number of iterations to run the optimization
        "optimizer": "adam",  # which optimizer to use for the iterative optimization
        "lr": 7.5e-2,  # learning rate of the optimizer
        "init": "measurement",  # initialization of the reconstruction before optimization
        "crop": 0,  # How much to crop out when considering the optimization loss
        "tv_reg": tv_reg,  # Total variation regularization parameter
        "l2_reg": l2_reg,  # L2 norm regularization parameter
        "l1_reg": l1_reg,  # L1 norm regularization parameter
        "plot_loss": False,  # Whether to plot the per-iteration loss during optimization
        "fraction": False,  # If true, computes fractional forward passes for lower memory consumption
        "upper_projection": False,  # If true, projects the image to [0,1] to prevent hot pixels from lowering contrast
    }
    def_opt_params.update(opt_params)

    if process:
        image = util.process(image, dim=image.shape[-2:], hot_pix=hot_pixel) * 0.9

    if not torch.is_tensor(image):
        image = torch.tensor(image)

    if warm_start is not None:
        warm_start = torch.tensor(warm_start)
        if warm_start.device is not device:
            warm_start = warm_start.to(device)

    recon = opt.image_recon(
        image.to(device).float(),
        psf_roft.to(device),
        model="lri",
        opt_params=def_opt_params,
        warm_start=warm_start,
        use_batch_conv=use_batch_conv,
        device=device,
        verbose=verbose,
    )

    return recon


def deeprd(
    image,
    seidel_coeffs,
    sharpness=1.25,
    model_path=None,
    process=True,
    hot_pixel=False,
    verbose=True,
    deconvolved=None,
    device=torch.device("cpu"),
):
    """

    Deblurs an image with a stack of PSFs using DeepRD.

    Parameters
    ----------
    image : np.ndarray or torch.Tensor
        The image to be deconvolved. Must be either (512,512) or (1024, 1024).

    seidel_coeffs : torch.Tensor
        The Seidel coefficient of the system

    sharpness : float, optional
        The sharpness parameter for the DeepRD model. Scales the Seidel coefficients to account for out-of-distribution data.
        Default is 1.5.

    model_path : str, optional
        The path to the pretrained DeepRD model. Default is
        "dl_models/pretrained/deeprd_22001".

    process : bool, optional
        Whether to process the image before deconvolution. Default is True.

    hot_pixel : bool, optional
        Whether to remove hot pixels from the image. Default is False.

    verbose : bool, optional
        Whether to print updates.

    deconvolved : np.ndarray or torch.Tensor, optional
        The standard deconvolved image for initialization. If not provided, it will be computed using the center Seidel PSF.

    device : torch.device, optional
        The device to use for the computation.

    Returns
    -------
    recon : torch.Tensor
        The deblurred image. Will be (N,N).

    """

    if process:
        image = util.process(image, dim=image.shape, hot_pix=hot_pixel) * 0.9

    if not torch.is_tensor(image):
        image = torch.tensor(image)
    if image.device is not device:
        image = image.to(device)

    if not torch.is_tensor(seidel_coeffs):
        seidel_coeffs = torch.tensor(seidel_coeffs)
    if seidel_coeffs.device is not device:
        seidel_coeffs = seidel_coeffs.to(device)

    if verbose:
        print("loading model...")

    model_seidelnet = DeepRD().to(device)

    if model_path is None:
        if image.shape[0] == 512:
            model_path = "dl_models/pretrained/deeprd_512"
            if verbose:
                print('Using pretrained model "deeprd_512"')

        elif image.shape[0] == 1024:
            model_path = "dl_models/pretrained/deeprd_1024"
            if verbose:
                print('Using pretrained model "deeprd_1024"')

        else:
            raise NotImplementedError("DeepRD model not found for this image size")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_path = os.path.join(current_dir, model_path)

    model = util.load_model(model_seidelnet, ckpt_path, device=device)

    if deconvolved is None:
        center_psf = get_psfs(
            seidel_coeffs=seidel_coeffs, dim=image.shape[0], model="lsi", device=device
        )

        deconvolved = deconvolve(image, center_psf, method="wiener", device=device)
        deconvolved = deconvolved.to(device)
    else:
        if not torch.is_tensor(deconvolved):
            deconvolved = torch.tensor(deconvolved)
        if deconvolved.device is not device:
            deconvolved = deconvolved.to(device)

    if verbose:
        print("deblurring...")

    input = torch.stack((image.float() - 0.5, deconvolved.float() - 0.5))
    output = torch.clip(model(input, sharpness * seidel_coeffs.T) + 0.5, 0, 1)
    recon = util.tensor_to_np(output)

    return recon


def deconvolve(
    image,
    psf,
    method="wiener",
    tv_reg=1e-9,
    l2_reg=1e-9,
    l1_reg=1e-9,
    iters=150,
    balance=3e-4,
    opt_params={},
    process=True,
    hot_pixel=False,
    verbose=True,
    device=torch.device("cpu"),
):
    """

    Deconvolves an image with a PSF.

    Parameters
    ----------
    image : np.ndarray or torch.Tensor
        The image to be deconvolved. Must be (M,N).

    psf : torch.Tensor
        The PSF to deconvolve the image with. Must be (M,N).

    method : str, optional
        The deconvolution method to use. Options are "wiener" and "iter".
        Default is "wiener".

    tv_reg : float, optional
        The TV regularization parameter. Default is 1e-10. (iter only)

    l2_reg : float, optional
        The L2 regularization parameter. Default is 1e-10. (iter only)

    balance : float, optional
        The balance parameter for the wiener filter. Default is 3e-4.

    opt_params : dict, optional
        The optimization/regularization parameters to use for deconvolution.
        See `opt.py` for details.

    process : bool, optional
        Whether to process the image before deconvolution. Default is True.

    verbose : bool, optional
        Whether to display a progress bar (only for 'iter')

    device : torch.device, optional
        The device to use for the computation.

    Returns
    -------
    recon : torch.Tensor
        The deconvolved image. Will be (M,N).

    Notes
    -----
    The "wiener" method is heavily based on the implementation in `scikit-image`.
    https://github.com/scikit-image/scikit-image/blob/v0.21.0/skimage/restoration/deconvolution.py#L10-L139

    """

    if len(psf.shape) != 2:
        raise ValueError("Deconvolution needs a single PSF")

    # default optimization parameters
    def_opt_params = {
        "iters": iters,  # number of iterations to run the optimization
        "optimizer": "adam",  # which optimizer to use for the iterative optimization
        "lr": 7.5e-2,  # learning rate of the optimizer
        "init": "measurement",  # initialization of the reconstruction before optimization
        "crop": 0,  # How much to crop out when considering the optimization loss
        "tv_reg": tv_reg,  # Total variation regularization parameter
        "l2_reg": l2_reg,  # L2 norm regularization parameter
        "l1_reg": l1_reg,  # L1 norm regularization parameter
        "plot_loss": False,  # Whether to plot the per-iteration loss during optimization
        "balance": balance,  # The balance parameter for the wiener filter
        "upper_projection": False,  # If true, projects the image to [0,1] to prevent hot pixels from lowering contrast
    }
    def_opt_params.update(opt_params)

    if process:
        image = util.process(image, dim=image.shape, hot_pix=hot_pixel) * 0.9

    if not torch.is_tensor(image):
        image = torch.tensor(image)
    if image.device is not device:
        image = image.to(device)

    if not torch.is_tensor(psf):
        psf = torch.tensor(psf)
    if psf.device is not device:
        psf = psf.to(device)

    if method == "wiener":
        recon = (
            (
                opt.wiener(
                    (image.float() - 0.5) * 2, psf, balance=def_opt_params["balance"]
                )
            )
            / 2
            + 0.5
        ).cpu()
    elif method == "iter":
        recon = opt.image_recon(
            image.float(),
            psf.float(),
            model="lsi",
            opt_params=def_opt_params,
            verbose=verbose,
            device=device,
        )

    else:
        raise ValueError("Invalid method")

    return recon


def blind(
    image,
    get_psf=False,
    balance=3e-4,
    iters=100,
    opt_params={},
    sys_params={},
    process=True,
    hot_pixel=False,
    verbose=True,
    device=torch.device("cpu"),
):
    """

    Blind deconvolves an image.

    Parameters
    ----------
    image : np.ndarray or torch.Tensor
        The image to be deconvolved. Must be (M,N).

    get_psf : bool, optional
        Whether to return the PSF. Default is False.

    balance : float, optional
        The balance parameter for the wiener filter. Default is 3e-4.

    iters : int, optional
        The number of iterations to run the optimization. Default is 100.

    opt_params : dict, optional
        The optimization/regularization parameters to use for deconvolution.

    sys_params : dict, optional
        The parameters to use for the optical system. See `seidel.py` for details.

    process : bool, optional
        Whether to process the image before deconvolution. Default is True.

    hot_pixel : bool, optional
        Whether to remove hot pixels from the image. Default is False.

    verbose : bool, optional
        Whether to display a progress bar (only for 'iter')

    device : torch.device, optional
        The device to use for the computation.

    Returns
    -------
    recon : torch.Tensor
        The deconvolved image. Will be (N,N).

    psf : torch.Tensor  (only if get_psf=True)
        The estimated PSF. Will be (N,N).

    """

    def_opt_params = {
        "iters": iters,  # number of iterations to run the optimization
        "optimizer": "adam",  # which optimizer to use for the iterative optimization
        "lr": 7.5e-2,  # learning rate of the optimizer
        "init": "measurement",  # initialization of the reconstruction before optimization
        "seidel_init": None,  # initialization of the seidel coefficients before optimization
        "crop": 0,  # How much to crop out when considering the optimization loss
        "balance": balance,  # The balance parameter for the wiener filter
        "plot_loss": False,  # Whether to plot the per-iteration loss during optimization
        "get_inter_seidels": False,  # Whether to return the intermediate seidel coefficients
        "get_inter_recons": False,  # Whether to return the intermediate reconstructions
    }
    def_opt_params.update(opt_params)

    def_sys_params = {
        "samples": min(image.shape),
        "L": 0,
        "lamb": 0.55e-6,
        "NA": 0.5,
    }
    radius_over_z = np.tan(np.arcsin(def_sys_params["NA"]))
    def_sys_params["L"] = ((min(image.shape)) * (def_sys_params["lamb"])) / (
        4 * (radius_over_z)
    )
    def_sys_params.update(sys_params)

    if process:
        image = util.process(image, dim=image.shape, hot_pix=hot_pixel) * 0.9

    if not torch.is_tensor(image):
        image = torch.tensor(image).float()
    if image.device is not device:
        image = image.to(device)

    recon, psf = opt.blind_recon(
        image,
        opt_params=def_opt_params,
        sys_params=def_sys_params,
        verbose=verbose,
        device=device,
    )

    if get_psf:
        return recon, psf
    else:
        return recon
