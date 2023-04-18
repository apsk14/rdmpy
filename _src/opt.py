# This file runs all the experiments

"""
Amit Kohli
10-18-20
This file runs all experiments and unit tests. It utilizes the code in src/ and experiment/
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import pathlib
from . import util, forward, seidel
from skimage.restoration import unwrap_phase as unwrap
import torch.fft as fft
from tqdm import tqdm
from kornia.filters.sobel import spatial_gradient

import pdb
import torch

dirname = str(pathlib.Path(__file__).parent.absolute())


# single shot version
def estimate_coeffs(
    calib_image, psf_list, sys_params, fit_params, device, show_psfs=False
):

    psfs_gt = torch.tensor(calib_image, device=device).float()
    if fit_params["seidel_init"] is not None:
        coeffs = torch.tensor(fit_params["seidel_init"], device=device)
    else:
        if fit_params["init"] == "zeros":
            coeffs = torch.zeros((fit_params["num_seidel"], 1), device=device)
        elif fit_params["init"] == "random":
            coeffs = torch.rand((fit_params["num_seidel"], 1), device=device)
        else:
            raise NotImplemented

    coeffs.requires_grad = True

    optimizer = torch.optim.Adam([coeffs], lr=fit_params["lr"])
    l2_loss_fn = torch.nn.MSELoss()
    l1_loss_fn = torch.nn.L1Loss()

    if fit_params["plot_loss"]:
        losses = []

    for iter in tqdm(range(fit_params["iters"])):
        # forward pass
        if coeffs.shape[0] < 6:
            psfs_estimate = seidel.compute_psfs(
                torch.cat((coeffs, torch.zeros(6 - coeffs.shape[0], 1, device=device))),
                desired_list=psf_list,
                stack=False,
                sys_params=sys_params,
                device=coeffs.device,
            )
        else:
            psfs_estimate = seidel.compute_psfs(
                coeffs,
                desired_list=psf_list,
                stack=False,
                sys_params=sys_params,
                device=coeffs.device,
            )
        # loss
        loss = l1_loss_fn(
            util.normalize(sum(psfs_estimate).float()), util.normalize(psfs_gt)
        ) + fit_params["reg"] * l2_loss_fn(coeffs, -coeffs)

        # if iter % 50 == 0:
        #     print(loss.item())
        #     # if iter % 100==0:
        #     psf_color = torch.stack(
        #         (
        #             util.normalize(sum(psfs_estimate).detach().cpu()),
        #             util.normalize(psfs_gt.detach().cpu()),
        #             torch.zeros_like(psfs_gt.detach().cpu()),
        #         ),
        #         dim=2,
        #     )
        #     psf_color = util.normalize(psf_color)
        #     color_crop = psf_color[250:-250, 250:-250, :]
        #     psf = sum(psfs_estimate).detach().cpu()
        #     print(coeffs)
        #     pdb.set_trace()

        if fit_params["plot_loss"]:
            losses += [loss.detach().cpu()]

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if show_psfs:
        psf_est = sum(psfs_estimate) / (len(psfs_estimate))

        plt.subplot(1, 2, 2)
        plt.tight_layout()
        plt.axis("off")
        plt.imshow(psf_est.detach().cpu(), cmap="inferno")
        plt.gca().set_title("Seidel PSFs")

        plt.subplot(1, 2, 1)
        plt.tight_layout()
        plt.axis("off")
        plt.imshow(psfs_gt.detach().cpu(), cmap="inferno")
        plt.gca().set_title("Measured PSFs")

    if fit_params["plot_loss"]:
        plt.figure()
        plt.plot(range(len(losses)), losses)
        plt.show()

    # util.show(torch.cat((psfs_gt/psfs_gt.max(), sum(psfs_estimate)/sum(psfs_estimate).max()), dim=1).detach().cpu())

    if coeffs.shape[0] < 6:
        return torch.cat((coeffs, torch.zeros(6 - coeffs.shape[0], 1, device=device)))
    else:
        return coeffs


def video_recon(
    measurement_stack, psf_data, model, opt_params, device, use_prev_frame=True
):

    num_frames = measurement_stack.shape[0]

    for i in range(num_frames):
        # recon frame by frame
        print("frame: " + str(i))
        curr_frame = measurement_stack[i, :, :]

        if use_prev_frame and i > 0:
            estimate = image_recon(
                curr_frame,
                psf_data,
                model,
                opt_params=opt_params,
                device=device,
                warm_start=measurement_stack[i - 1, :, :],
            )
        else:
            estimate = image_recon(
                curr_frame, psf_data, model, opt_params=opt_params, device=device
            )

        # for memory efficiency, replace measurement stack
        measurement_stack[i, :, :] = estimate
    return measurement_stack


def image_recon(
    measurement, psf_data, model, opt_params, device, verbose=False, warm_start=None
):

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

    loss_fn = torch.nn.MSELoss()
    crop = opt_params["crop"]

    losses = []

    # if model == "lri":
    #     reg_vec = (torch.linspace(0, 1, measurement.shape[0])[None, :]) * opt_params[
    #         "l2_reg"
    #     ]
    #     reg_vec = reg_vec.to(device)

    if opt_params["plot_loss"]:
        losses = []

    for it in tqdm(range(opt_params["iters"])):

        # forward pass and loss
        if model == "lsi":
            measurement_guess = forward.padded_lsi(estimate, psf_data)
            if crop > 0:
                loss = (
                    loss_fn(
                        (measurement_guess)[crop:-crop, crop:-crop],
                        (measurement)[crop:-crop, crop:-crop],
                    )
                    + tv(estimate[crop:-crop, crop:-crop], opt_params["tv_reg"])
                    + opt_params["l2_reg"] * torch.norm(estimate)
                )
            else:
                loss = (
                    loss_fn(measurement_guess, measurement)
                    + tv(estimate, opt_params["tv_reg"])
                    + opt_params["l2_reg"] * torch.norm(estimate)
                )
        else:
            measurement_guess, obj = forward.lri(
                estimate, psf_data, method="normal", device=device, verbose=False
            )
            if crop > 0:
                loss = (
                    loss_fn(
                        (measurement_guess)[crop:-crop, crop:-crop],
                        (measurement)[crop:-crop, crop:-crop],
                    )
                    + tv(estimate[crop:-crop, crop:-crop], opt_params["tv_reg"])
                    + opt_params["l2_reg"] * torch.norm(estimate)
                )
            else:
                loss = (
                    loss_fn(measurement_guess, measurement)
                    + tv(estimate, opt_params["tv_reg"])
                    + opt_params["l2_reg"] * torch.norm(estimate)
                )

        if opt_params["plot_loss"]:
            losses += [loss.detach().cpu()]

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # project onto [0,1]
        estimate.data[estimate.data < 0] = 0
        estimate.data[estimate.data > 1] = 1

    if opt_params["plot_loss"]:
        plt.figure()
        plt.plot(range(len(losses)), losses)
        plt.show()

    return util.normalize(estimate.detach().cpu().float().numpy())


def blind_recon_gradient(
    measurement, opt_params, sys_params, device=torch.device("cpu")
):

    if opt_params["seidel_init"] is None:
        coeffs = torch.zeros((1, 1), device=device)

    coeffs.requires_grad = True

    optimizer = torch.optim.Adam([coeffs], lr=opt_params["lr"])

    if opt_params["plot_loss"]:
        losses = []

    for iter in tqdm(range(opt_params["iters"])):
        # forward pass
        psfs_estimate = seidel.compute_psfs(
            torch.cat((coeffs, torch.zeros(5, 1, device=device))),
            desired_list=[(0, 0)],
            sys_params=sys_params,
            device=coeffs.device,
        )[0]
        recon = wiener_torch(
            ((measurement - 0.5) * 2), psfs_estimate, balance=opt_params["balance"]
        )  # can also use .wiener with ,balance=3e-4
        recon = (recon / 2) + 0.5  # back-scale

        recon_forward = forward.padded_lsi(recon, psfs_estimate)
        data_loss = torch.mean((recon_forward - measurement) ** 2)

        # loss (maximizing acutance)
        sharpness_loss = -torch.mean(
            torch.abs(spatial_gradient(recon[None, None, :, :], mode="diff"))
        )
        loss = sharpness_loss

        # pdb.set_trace()
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

    return (
        recon.detach().cpu().numpy(),
        psfs_estimate.detach().cpu().numpy(),
        coeffs.detach().cpu().numpy(),
    )


def blind_recon_grid(measurement, opt_params, sys_params, device=torch.device("cpu")):

    # coarse pass
    loss_list = []
    for coeff in torch.arange(0, 10, 1):
        coeffs = torch.ones((1, 1), device=device) * coeff
        # forward pass
        psfs_estimate = seidel.compute_psfs(
            torch.cat((coeffs, torch.zeros(5, 1, device=device))),
            desired_list=[(0, 0)],
            sys_params=sys_params,
            device=coeffs.device,
        )[0]
        recon = wiener_torch(
            ((measurement - 0.5) * 2), psfs_estimate, balance=opt_params["balance"]
        )  # can also use .wiener with ,balance=3e-4
        recon = (recon / 2) + 0.5  # back-scale

        recon_forward = forward.padded_lsi(recon, psfs_estimate)
        loss = torch.mean((recon_forward - measurement) ** 2)
        loss_list += [loss.cpu()]

    min_loss_idx = np.argmin(loss_list)
    pdb.set_trace()

    if min_loss_idx == 0:
        start = 0
        end = 1
    else:
        start = min_loss_idx - -0.5
        end = min_loss_idx + 0.5

    # fine pass
    for coeff in tqdm(torch.linspace(start=start, end=end, steps=opt_params["iters"])):
        coeffs = torch.ones((1, 1), device=device) * coeff
        # forward pass
        psfs_estimate = seidel.compute_psfs(
            torch.cat((coeffs, torch.zeros(5, 1, device=device))),
            desired_list=[(0, 0)],
            sys_params=sys_params,
            device=coeffs.device,
        )[0]
        recon = wiener_torch(
            ((measurement - 0.5) * 2), psfs_estimate, balance=opt_params["balance"]
        )
        recon = (recon / 2) + 0.5  # back-scale
        recon_forward = forward.padded_lsi(recon, psfs_estimate)
        loss = torch.mean((recon_forward - measurement) ** 2)
        loss_list += [loss.cpu()]

    coeff = np.argmin(loss_list)
    coeffs = torch.ones((1, 1), device=device) * coeff
    psfs_estimate = seidel.compute_psfs(
        torch.cat((coeffs, torch.zeros(5, 1, device=device))),
        desired_list=[(0, 0)],
        sys_params=sys_params,
        device=coeffs.device,
    )[0]
    recon = wiener_torch(
        ((measurement - 0.5) * 2), psfs_estimate, balance=opt_params["balance"]
    )
    pdb.set_trace()

    return (
        recon.detach().cpu().numpy(),
        psfs_estimate.detach().cpu().numpy(),
        coeffs.detach().cpu().numpy(),
    )


def tv(img, weight):
    tv_h = ((img[1:, :] - img[:-1, :]).abs()).sum()
    tv_w = ((img[:, 1:] - img[:, :-1]).abs()).sum()
    return weight * (tv_h + tv_w)


def center_crop(measurement, des_shape):
    # Center crop
    m_center = (measurement.shape[0] // 2, measurement.shape[1] // 2)
    left, right, up, down = (
        m_center[1] - des_shape[1] // 2,
        m_center[1] + int(np.round(des_shape[1] / 2)),
        m_center[0] - des_shape[0] // 2,
        m_center[0] + int(np.round(des_shape[0] / 2)),
    )
    # TODO: Debug this for images of an odd size.
    measurement = measurement[left:right, up:down]
    return measurement


def blind_recon_alt(measurement, opt_params, sys_params, device):

    if opt_params["seidel_init"] is None:
        # coeffs = torch.rand((1, 1), device=device)
        coeffs = torch.zeros((1, 1), device=device)

    # estimate = torch.tensor(measurement, device=device).float()
    estimate = torch.zeros_like(measurement, device=device)
    measurement = torch.tensor(measurement, device=device).float()

    coeffs.requires_grad = True
    estimate.requires_grad = True

    opt_est = torch.optim.Adam([estimate], lr=opt_params["lr"])
    opt_coeff = torch.optim.Adam([coeffs], lr=opt_params["lr"])
    l2_loss_fn = torch.nn.MSELoss()
    # fig = plt.figure()
    # camera = Camera(fig)
    SWITCH_FREQ = 10
    est = True

    for iter in tqdm(range(opt_params["iters"])):
        # forward pass
        # psfs_estimate, pupils = seidel.compute_psfs(torch.cat((torch.zeros(1, 1, device=device), coeffs, torch.zeros(4, 1, device=device) )), desired_list=[(0,0)], sys_params=sys_params, device=coeffs.device)
        psfs_estimate = seidel.compute_psfs(
            torch.cat(
                (
                    torch.zeros(1, 1, device=device),
                    coeffs,
                    torch.zeros(4, 1, device=device),
                )
            ),
            desired_list=[(0, 0)],
            sys_params=sys_params,
            device=coeffs.device,
        )[0]
        measurement_guess = forward.lsi(estimate, psfs_estimate)

        # loss
        sharpness = -torch.norm(
            torch.abs(spatial_gradient(estimate[None, None, :, :], mode="diff"))
        )
        consistency = (
            l2_loss_fn(measurement_guess.float(), measurement)
            + tv(estimate, opt_params["tv_reg"])
            + opt_params["l2_reg"] * torch.norm(estimate)
        )

        loss = consistency + 1e-8 * sharpness

        # print(str(iter) + ' consistency: ' + str(consistency.detach().cpu().numpy()) + ', sharpness: ' + str(-sharpness.detach().cpu().numpy()))

        # backward
        opt_est.zero_grad()
        opt_coeff.zero_grad()
        loss.backward()

        # if est:
        opt_est.step()
        # else:
        opt_coeff.step()

        estimate.data[estimate.data < 0] = 0
        estimate.data[estimate.data > 1] = 1
        if iter % SWITCH_FREQ:
            est = not est

    # print(coeffs)
    # util.show(estimate.detach().cpu())

    # torch.save(psfs_estimate[0], cache_path+'psf_fit.pt')
    return (
        estimate.detach().cpu().numpy(),
        psfs_estimate.detach().cpu().numpy(),
        coeffs.detach().cpu().numpy(),
    )


def wiener_torch(image, psf, balance=3e-4, reg=None, is_real=True, clip=True):
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


def laplacian(ndim, shape, is_real=True):
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


def image_quad_norm(image):
    np.sum(np.sum(np.abs(image) ** 2, axis=-1), axis=-1)
