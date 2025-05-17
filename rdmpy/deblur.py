import pathlib
import os

import torch
import numpy as np

from ._src import opt, util
from .calibrate import get_rdm_psfs
from .dl_models.DeepRD.DeepRD import UNet as DeepRD

dirname = str(pathlib.Path(__file__).parent.absolute())


def ring_deconvolve(
    image,
    psf_roft,
    patch_size=0,
    iters=150,
    lr=5e-2,
    tv_reg=1e-10,
    l2_reg=1e-10,
    l1_reg=0,
    opt_params={},
    warm_start=None,
    process=True,
    hot_pixel=False,
    verbose=True,
    device=torch.device("cpu"),
):
    """Ring deconvolve an image.

    This function deblurs a centered, square image from a rotationally symmetric imaging system. It uses a
    stack of PSFs, one for each ring in the image. The PSFs should be in the rotational Fourier domain.

    Parameters
    ----------
    image : np.ndarray or torch.Tensor
        The image to be deconvolved. Must be (N,N) and centered.

    psf_roft : torch.Tensor
        The stack of PSFs to deconvolve the image with. The PSFs should be in the Rotational Fourier domain.
        Will be (L, M, L) where L is the number of PSFs/radii and M is the number of angles. See `rdmpy.calibrate.get_rdm_psfs` for details.

    patch_size : int
        The radial size of the rings to use for ring convolution. If 0, deconvolution is ring by ring.

    iters : int
        The number of iterations to run the optimization.

    lr : float
        The learning rate of the optimizer.

    tv_reg : float
        The Total Variation regularization parameter.

    l2_reg : float
        The L2 norm regularization parameter.

    l1_reg : float
        The L1 norm regularization parameter.

    opt_params : dict
        Ad optimization/regularization parameters to use for deconvolution.
        See `opt.py` for details.

    warm_start : np.ndarray or torch.Tensor
        The warm start for the optimization. If not provided, it will be initialized to `image`.

    process : bool
        Whether to process the image before deconvolution.

    hot_pixel : bool
        Whether to remove hot pixels from the image.

    verbose : bool
        Whether to display a progress bar.

    device : torch.device
        The device to use for the computation.

    Returns
    -------
    recon : torch.Tensor
        The ring deconvolved image.

    Notes
    -----
    Since this method takes advantage of rotationally symmetry, the input images must be square and
    centered on the optical axis; for non-square inputs, one can either pad or crop the image to make it square.
    Like other deconvolution methods, ring deconvolution is prone to ringing artifacts when
    PSFs are large and there is significant noise. They often manifest as circles radiating out
    from the center of the image. They can be reduced through regularization, especially if
    the object is smooth or sparse.
    """

    if len(psf_roft.shape) != 3:
        raise ValueError("Ring deconvolution needs a radial stack of PSF RoFTs")

    if patch_size < 0 or patch_size > image.shape[0]:
        raise AssertionError(
            f"Patch size {patch_size} must be between 0 and {image.shape[0]}"
        )

    # default optimization parameters
    def_opt_params = {
        "iters": iters,  # number of iterations to run the optimization
        "optimizer": "adam",  # which optimizer to use for the iterative optimization
        "lr": lr,  # learning rate of the optimizer
        "init": "measurement",  # initialization of the reconstruction before optimization
        "patch_size": patch_size,  # size of the patches to use for ring convolution
        "crop": 0,  # How much to crop out when considering the optimization loss
        "tv_reg": tv_reg,  # Total variation regularization parameter
        "l2_reg": l2_reg,  # L2 norm regularization parameter
        "l1_reg": l1_reg,  # L1 norm regularization parameter
        "plot_loss": False,  # Whether to plot the per-iteration loss during optimization
        "upper_projection": False,  # If true, projects the image to [0,1] to prevent hot pixels from lowering contrast
    }
    def_opt_params.update(opt_params)

    if process:
        image = util.process(image, dim=image.shape[-2:], hot_pix=hot_pixel) * 0.9

    if not torch.is_tensor(image):
        image = torch.tensor(image)
    if image.device is not device:
        image = image.to(device)

    if psf_roft.device is not device:
        psf_roft = psf_roft.to(device)

    if warm_start is not None:
        warm_start = torch.tensor(warm_start)
        if warm_start.device is not device:
            warm_start = warm_start.to(device)

    recon = opt.image_recon(
        image.float(),
        psf_roft,
        model="lri_patch" if patch_size > 0 else "lri",
        opt_params=def_opt_params,
        warm_start=warm_start,
        device=device,
        verbose=verbose,
    )

    return recon


def sheet_deconvolve(
    image_stack,
    psf_stack,
    iters=150,
    lr=1e-3,
    tv_reg=0,
    l2_reg=0,
    l1_reg=0,
    opt_params={},
    warm_start=None,
    process=True,
    hot_pixel=False,
    verbose=True,
    device=torch.device("cpu"),
):
    r"""Sheet deconvolve a volume.

    This function deblurs a 3D volume imaged with a light sheet microscope, where the 0th axis is
    the axis along which the light sheet is focused (i.e., spatially varying axis). It uses a
    list of 3D PSFs, one at each location along the light sheet axis. Assuming symmetry of the light
    sheet, only half the PSF locations are needed. Thus, len(psf_stack) = image_stack.shape[0] // 2.


    Parameters
    ----------
    image_stack : np.ndarray or torch.Tensor
        The volume to be deconvolved. Must be (M,N,L) where M is even.

    psf_stack : list of torch.Tensors
        The stack of light sheet PSFs to deconvolve the image with.
        Must be a list of PSFs of length M//2.

    iters : int,
        The number of iterations to run the optimization.

    lr : float,
        The learning rate of the optimizer.

    tv_reg : float,
        The Total Variation regularization parameter.

    l2_reg : float,
        The L2 norm regularization parameter.

    l1_reg : float,
        The L1 norm regularization parameter.

    opt_params : dict,
        Optimization/regularization parameters to use for iterative deconvolution. See 'rdmpy.deblur.ring_deconvolve'
        for details.

    warm_start : np.ndarray or torch.Tensor,
        The warm start for the optimization. If not provided, it will be initialized to `image_stack`.

    process : bool,
        Whether to process the image before deconvolution.

    hot_pixel : bool,
        Whether to remove hot pixels from the image.

    verbose : bool,
        Whether to display a progress bar.

    device : torch.device,
        The device to use for the computation.

    Returns
    -------
    recon : torch.Tensor
        The ring deconvolved image.

    Notes
    -----
    Like other deconvolution methods, sheet deconvolution is prone to ringing artifacts when
    PSFs are large and there is significant noise. They often manifest as lines perpendicular
    to the spatially varying axis. They can be reduced through regularization, especially if
    the object is smooth or sparse.
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
        "upper_projection": False,  # If true, projects the image to [0,1] to prevent hot pixels from lowering contrast
    }
    def_opt_params.update(opt_params)

    if hot_pixel:
        _, image_stack = torch.from_numpy(
            util.find_outlier_pixels(image_stack.cpu().numpy(), tolerance=0.5),
            device=device,
        )
    if process:
        image_stack = util.normalize(image_stack) * 0.9

    # TODO assert that PSF is the right shape

    if not torch.is_tensor(image_stack):
        image_stack = torch.tensor(image_stack)

    image_stack = image_stack.to(device)

    if warm_start is not None:
        warm_start = torch.tensor(warm_start)
        if warm_start.device is not device:
            warm_start = warm_start.to(device)

    recon = opt.volume_recon(
        image_stack,
        psf_stack,
        opt_params=def_opt_params,
        warm_start=warm_start,
        verbose=verbose,
        device=device,
    )

    # TODO: Add additional optimization methods
    # if method == "autograd":
    #     recon = opt.volume_recon(
    #         image_stack,
    #         psf_stack,
    #         opt_params=def_opt_params,
    #         warm_start=warm_start,
    #         device=device,
    #         verbose=verbose,
    #     )
    # elif method == "manual":
    #     recon = opt.volume_recon_manual(
    #         image_stack,
    #         psf_stack,
    #         opt_params=def_opt_params,
    #         warm_start=warm_start,
    #         device=device,
    #         verbose=verbose,
    #     )
    # elif method == "cg":
    #     recon = opt.volume_recon_cg_normal(
    #         image_stack,
    #         psf_stack,
    #         opt_params=def_opt_params,
    #         warm_start=warm_start,
    #         preconditioner="jacobi",
    #         center_vals=center_vals,
    #         device=device,
    #         verbose=verbose,
    #     )
    # elif method == "scipy":
    #     recon = opt.volume_recon_scipy(
    #         image_stack, psf_stack, opt_params, device=device
    #     )

    return recon


def deeprd(
    image,
    seidel_coeffs,
    sharpness=1.25,
    model_path=None,
    noisy=False,
    warm_start=None,
    process=True,
    hot_pixel=False,
    verbose=True,
    device=torch.device("cpu"),
):
    """Deblur an image using DeepRD.

    Deep ring deconvolution is a deep-learning version of ring deconvolution. It uses a
    a hypernetwork to map Seidel coefficients to a coefficient-specific deblurring neural network.
    Then the deblurring network is used to deblur the input image.

    Parameters
    ----------
    image : np.ndarray or torch.Tensor
        The image to be deconvolved. Must be either (512,512) or (1024, 1024).

    seidel_coeffs : torch.Tensor
        The 6 Seidel coefficient of the system including defocus. Format is [sphere, coma, astigmatism, field curvature, distortion, defocus].

    sharpness : float
        The sharpness parameter for the DeepRD model. Scales the Seidel coefficients to account for out-of-distribution data.

    model_path : str
        The path to the pretrained DeepRD model. Will default to our pretrained models.

    noisy : bool
        Whether to remove noise from the image via thresholding. Use when there are significant noise-related artifacts.

    warm_start : np.ndarray or torch.Tensor
        The warm start initialization for the model. If not provided, it will be initialized to input image deconvolved with the center Seidel PSF.

    process : bool
        Whether to process the image before deconvolution. Default is True.

    hot_pixel : bool
        Whether to remove hot pixels from the image. Default is False.

    verbose : bool
        Whether to print updates.

    device : torch.device
        The device to use for the computation.

    Returns
    -------
    recon : torch.Tensor
        The deblurred image.

    Notes
    -----
    DeepRD, like other deep learning models, is subject to unpredicatble errors/artifacts in the presence of
    out-of-distribution data. It is recommended to use the model on data that is similar to the training data or
    finetune the model on your data.
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
        elif image.shape[0] == 1024:
            model_path = "dl_models/pretrained/deeprd_1024"
        else:
            # TODO: Add support for other image sizes
            raise ValueError("Image size not supported")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_path = os.path.join(current_dir, model_path)

    model = util.load_model(model_seidelnet, ckpt_path, device=device)

    if warm_start is None:
        center_psf = get_rdm_psfs(
            seidel_coeffs=seidel_coeffs, dim=image.shape[0], model="lsi", device=device
        )

        warm_start = deconvolve(image, center_psf, method="wiener", device=device)
        warm_start = warm_start.to(device)
    else:
        if not torch.is_tensor(warm_start):
            warm_start = torch.tensor(warm_start)
        if warm_start.device is not device:
            warm_start = warm_start.to(device)

    if verbose:
        print("deblurring...")

    input = torch.stack((image.float() - 0.5, warm_start.float() - 0.5))
    output = torch.clip(model(input, sharpness * seidel_coeffs.T) + 0.5, 0, 1)
    recon = util.tensor_to_np(output)

    if noisy:
        recon[recon < np.quantile(recon, noisy)] = 0

    return recon


def deconvolve(
    image,
    psf,
    method="wiener",
    iters=150,
    lr=7.5e-2,
    tv_reg=1e-9,
    l2_reg=1e-9,
    l1_reg=1e-9,
    balance=3e-4,
    opt_params={},
    process=True,
    hot_pixel=False,
    verbose=True,
    device=torch.device("cpu"),
):
    """Deconvolve an image with a PSF.

    Parameters
    ----------
    image : np.ndarray or torch.Tensor
        The image to be deconvolved. Must be (M,N).

    psf : torch.Tensor
        The PSF to deconvolve the image with. Must be (M,N).

    method : str
        The deconvolution method to use. Options are "wiener" and "iter".
        Default is "wiener".

    iters : int
        The number of iterations to run the optimization. Default is 150.

    lr : float
        The learning rate of the optimizer. Default is 7.5e-2.

    tv_reg : float
        The TV regularization parameter. Default is 1e-10. (iter only)

    l2_reg : float
        The L2 regularization parameter. Default is 1e-10. (iter only)

    balance : float
        The balance parameter for the wiener filter. Default is 3e-4.

    opt_params : dict
        The optimization/regularization parameters to use for deconvolution.
        See `opt.py` for details.

    process : bool,
        Whether to process the image before deconvolution. Default is True.

    verbose : bool
        Whether to display a progress bar (only for 'iter')

    device : torch.device
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
        "lr": lr,  # learning rate of the optimizer
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
    iters=100,
    lr=7.5e-2,
    balance=3e-4,
    opt_params={},
    sys_params={},
    process=True,
    hot_pixel=False,
    verbose=True,
    device=torch.device("cpu"),
):
    """Deconvolve an image with no PSF.

    Parameters
    ----------
    image : np.ndarray or torch.Tensor
        The image to be deconvolved.

    get_psf : bool
        Whether to return the PSF.

    iters : int
        The number of iterations to run the optimization.

    lr : float
        The learning rate of the optimizer.

    balance : float
        The balance parameter for the wiener filter.

    opt_params : dict
        The optimization/regularization parameters to use for deconvolution.

    sys_params : dict
        The parameters to use for the optical system. See `rdmpy.calibrate` for details.

    process : bool
        Whether to process the image before deconvolution. Default is True.

    hot_pixel : bool
        Whether to remove hot pixels from the image. Default is False.

    verbose : bool
        Whether to display a progress bar (only for 'iter')

    device : torch.device
        The device to use for the computation.

    Returns
    -------
    recon : torch.Tensor
        The deconvolved image.

    psf : torch.Tensor  (only if get_psf=True)
        The estimated PSF.

    Notes
    -----
    The blind deconvolution uses a heuristic measure of image sharpness as a loss function.
    Thus, it is possible when there is sufficient noise that the PSF will be estimated to be
    overly large and thus cause an over-sharping of the image. In such cases, reduce the number of interations
    to effecively regularize the optimization.
    """

    def_opt_params = {
        "iters": iters,  # number of iterations to run the optimization
        "optimizer": "adam",  # which optimizer to use for the iterative optimization
        "lr": lr,  # learning rate of the optimizer
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
