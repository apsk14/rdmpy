#This file runs all the experiments

"""
Amit Kohli
10-18-20
This file runs all experiments and unit tests. It utilizes the code in src/ and experiment/
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import pathlib
from . import util, lri, seidel
from skimage.restoration import unwrap_phase as unwrap
import torch.fft as fft
from tqdm import tqdm

import pdb
import torch
dirname = str(pathlib.Path(__file__).parent.absolute())

# TODO: Make an experiments/ directory and have different files for each experiment.
# TODO: BUG: wiener doesn't work unless 1) run forward with inverse flips, 2 delete inverse, 3) run inverse without flips


def video_recon(measurement_stack, psf_stack_roft, opt_params, device):
    num_frames = measurement_stack.shape[0]
    psfs = None
    for i in range(num_frames):
        print('frame: ' + str(i))
        curr_frame = measurement_stack[i, :, :]
        estimate, _, _, psfs = image_recon(curr_frame, psf_stack_roft, opt_params, device)
        # for memory recons, replace measurement stack
        measurement_stack[i, :, :] = estimate
    return measurement_stack


def image_recon(measurement, psf_stack_roft, opt_params, diff, device):
    dim = measurement.shape
    if opt_params['init'] == 'measurement':
        estimate = torch.tensor(measurement, device=device).float()
    elif opt_params['init'] == 'zero':
        estimate = torch.zeros(dim, device=device)
    elif opt_params['init'] == 'noise':
        estimate = torch.randn(dim, device=device)
    else:
        raise NotImplemented
    measurement = torch.tensor(measurement, device=device).float()

    estimate.requires_grad = True

    if opt_params['optimizer'] == 'adam':
        optimizer = torch.optim.Adam([estimate], lr=opt_params['lr'])
    elif opt_params['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD([estimate], lr=opt_params['lr'])
    else:
        raise NotImplemented

    loss_fn = torch.nn.MSELoss()
    crop = opt_params['crop']

    losses = []
    for it in tqdm(range(opt_params['iters'])):

        # forward pass
        measurement_guess = lri.blur(estimate, psf_stack_roft, method='normal', device=device, verbose=False, diff=diff)

        # loss
        if crop > 0:
            loss = loss_fn((measurement_guess)[crop:-crop, crop:-crop], (measurement)[crop:-crop, crop:-crop]) + \
                    tv(estimate[crop:-crop, crop:-crop], opt_params['reg'])
        else:
            loss = loss_fn(measurement_guess, measurement) + tv(estimate, opt_params['reg'])

        losses += [loss.detach().cpu()]

        # print loss
        #print(it, loss.item())

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # project onto [0,1]
        estimate.data[estimate.data < 0] = 0
        estimate.data[estimate.data > 1] = 1

    return util.normalize(estimate.detach().cpu().float().numpy())



def estimate_coeffs(psf_data, sys_params, opt_params, std_init='random', init=None, plot=False, device=torch.device('cpu')):

    cache_path = dirname + '/data/.cache/'
    half_length = sys_params['L'] / 2
    psf_list = [(torch.tensor(i[0], device=device).float(), torch.tensor(i[1], device=device).float()) for i in psf_data[0]]
    psfs_gt = torch.stack([torch.tensor(psf, device=device).float() for psf in psf_data[1]], dim=0)
    if init is None:
        if std_init == 'zeros':
            coeffs = torch.zeros((3, 1), device=device)
        elif std_init == 'random':
            coeffs = torch.rand((3, 1), device=device)
        else:
            raise NotImplemented
    else:
        coeffs = torch.tensor(init, device=device).float()

    coeffs.requires_grad = True

    optimizer = torch.optim.Adam([coeffs], lr=opt_params['lr'])
    #optimizer = torch.optim.SGD([coeffs], lr=opt_params['lr'])
    l1_loss_fn = torch.nn.L1Loss()
    l2_loss_fn = torch.nn.MSELoss()
    smooth_l1 = torch.nn.SmoothL1Loss()
    fig = plt.figure()
    camera = Camera(fig)
    crop = 225

    for iter in range(opt_params['iters']):
        # forward pass
        psfs_estimate, pupils = seidel.compute_psfs(torch.cat((torch.zeros(1, 1, device=device), coeffs, torch.zeros(2, 1, device=device) )), desired_list=psf_list, sys_params=sys_params, device=coeffs.device)\
        # loss
        loss = l2_loss_fn(torch.stack(psfs_estimate, dim=0).float(), psfs_gt) + opt_params['reg']*l2_loss_fn(coeffs, -coeffs)
        # loss = loss_fn((measurement[:,:350]), (measurement_guess[:,:350])) #+ tv(estimate, 1e-9)
        print(iter, loss.item())

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    print(coeffs)
    if plot:
        for gt, psf, pupil in zip(psfs_gt, psfs_estimate, pupils):
            #pdb.set_trace()
            util.show(torch.cat((gt/gt.max(), psf/psf.max()), dim=1).detach())
            util.show(psf.detach())
            plt.figure()
            plt.imshow(unwrap(torch.angle(pupil).detach()))
            plt.colorbar()
            plt.show()


    #torch.save(psfs_estimate[0], cache_path+'psf_fit.pt')
    return torch.cat((torch.zeros(1, 1, device=device), coeffs, torch.zeros(2, 1, device=device)), ).detach()

# single shot version
def estimate_coeffs_ss(psf_img, psf_list, sys_params, opt_params, device, std_init='zeros', init=None, plot=False):

    #psf_list = [(torch.tensor(i[0], device=device).float(), torch.tensor(i[1], device=device).float()) for i in psf_list]
    psfs_gt = torch.tensor(psf_img, device=device).float()
    if init is None:
        if std_init == 'zeros':
            coeffs = torch.zeros((3, 1), device=device)
        elif std_init == 'random':
            coeffs = torch.rand((3, 1), device=device)
        else:
            raise NotImplemented
    else:
        coeffs = torch.tensor(init, device=device).float()

    coeffs.requires_grad = True

    optimizer = torch.optim.Adam([coeffs], lr=opt_params['lr'])
    #optimizer = torch.optim.SGD([coeffs], lr=opt_params['lr'])
    l1_loss_fn = torch.nn.L1Loss()
    l2_loss_fn = torch.nn.MSELoss()
    smooth_l1 = torch.nn.SmoothL1Loss()
    fig = plt.figure()

    for iter in tqdm(range(opt_params['iters'])):
        # forward pass
        psfs_estimate = seidel.compute_psfs(torch.cat((torch.zeros(1, 1, device=device), coeffs, torch.zeros(2, 1, device=device) )), desired_list=psf_list, stack=False, sys_params=sys_params, device=coeffs.device)\
        # loss
        loss = l2_loss_fn(sum(psfs_estimate).float(), psfs_gt) + opt_params['reg']*l2_loss_fn(coeffs, -coeffs)
        # loss = loss_fn((measurement[:,:350]), (measurement_guess[:,:350])) #+ tv(estimate, 1e-9)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    if plot:
        util.show(torch.cat((psfs_gt/psfs_gt.max(), sum(psfs_estimate)/sum(psfs_estimate).max()), dim=1).detach().cpu())
        util.show(sum(psfs_estimate).detach().cpu())


    #torch.save(psfs_estimate[0], cache_path+'psf_fit.pt')
    return torch.cat((torch.zeros(1, 1, device=device), coeffs, torch.zeros(2, 1, device=device)), ).detach()


def blind_image_recon(measurement, method, model, opt_params, cache, init=None):

    device = model.dataset.device
    if init is None:
        if method == 'lsi_blind':
            coeffs = torch.randn((1, 1), device=device)
        elif method == 'lri_blind':
            coeffs = torch.randn((3, 1), device=device)

    estimate = torch.tensor(measurement, device=device).float()
    #estimate = torch.zeros(model.dataset.dim, device=device)
    measurement = torch.tensor(measurement, device=device).float()
    

    coeffs.requires_grad = True
    estimate.requires_grad = True

    sys_params = model.dataset.sys_params

    opt_est = torch.optim.Adam([estimate], lr=opt_params['lr'])
    opt_coeff = torch.optim.Adam([coeffs], lr=opt_params['lr'])
    #optimizer = torch.optim.SGD([coeffs], lr=opt_params['lr'])
    l1_loss_fn = torch.nn.L1Loss()
    l2_loss_fn = torch.nn.MSELoss()
    smooth_l1 = torch.nn.SmoothL1Loss()
    fig = plt.figure()
    camera = Camera(fig)
    crop = 225
    COUNT = 10
    est = True

    for iter in range(opt_params['iters']):
        # forward pass
        #psfs_estimate, pupils = seidel.compute_psfs(torch.cat((torch.zeros(1, 1, device=device), coeffs, torch.zeros(4, 1, device=device) )), desired_list=[(0,0)], sys_params=sys_params, device=coeffs.device) 
        measurement_guess,_ = model.forward(estimate, method=method, coeffs=coeffs, cache=cache, verbose=True)

        # loss
        sharpness = -torch.norm(torch.abs(kornia.spatial_gradient(estimate[None, None, :, :], mode='diff')))
        consistency = l2_loss_fn(measurement_guess.float(), measurement) + tv(estimate, opt_params['reg'])

        loss =  consistency + tv(estimate, opt_params['reg']) + 1e-10*sharpness

        print(str(iter) + ' consistency: ' + str(consistency.detach().cpu().numpy()) + ', sharpness: ' + str(-sharpness.detach().cpu().numpy()))

        # backward
        opt_est.zero_grad()
        opt_coeff.zero_grad()
        loss.backward()

        if est:
           opt_est.step() 
        else:
           opt_coeff.step() 

        estimate.data[estimate.data < 0] = 0
        estimate.data[estimate.data > 1] = 1
        if iter % COUNT:
            est = not est

    print(coeffs)
    util.show(estimate.detach().cpu())

    #torch.save(psfs_estimate[0], cache_path+'psf_fit.pt')
    return estimate.detach().cpu().numpy(), coeffs.detach().cpu().numpy()


def blind_recon(measurement, method, model, opt_params, cache, init=None):

    device = model.dataset.device

    if init is None:
        coeffs = torch.zeros((1, 1), device=device) 

    measurement = torch.tensor(measurement, device=device).float()

    coeffs.requires_grad = True
    sys_params = model.dataset.sys_params

    optimizer = torch.optim.Adam([coeffs], lr=opt_params['lr'])
    #optimizer = torch.optim.SGD([coeffs], lr=opt_params['lr'])
    l1_loss_fn = torch.nn.L1Loss()
    l2_loss_fn = torch.nn.MSELoss()
    smooth_l1 = torch.nn.SmoothL1Loss()
    fig = plt.figure()
    camera = Camera(fig)
    crop = 225

    for iter in range(opt_params['iters']):
        # forward pass
        psfs_estimate, _ = seidel.compute_psfs(torch.cat((torch.zeros(1, 1, device=device), coeffs, torch.zeros(4, 1, device=device) )), desired_list=[(0,0)], sys_params=sys_params, device=coeffs.device) 
        recon = wiener_torch(((measurement - 0.5) * 2).reshape(model.dataset.dim), psfs_estimate[0]) # can also use .wiener with ,balance=3e-4
        recon = (recon/2)+0.5 # back-scale

        # loss (maximizing acutance)
        loss = -torch.mean(torch.abs(kornia.spatial_gradient(recon[None, None, :, :], mode='diff')))
        print(iter, -loss.item())

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #coeffs.data[coeffs.data < 0] = 0

    print(coeffs)

    #torch.save(psfs_estimate[0], cache_path+'psf_fit.pt')
    return recon.detach().cpu().numpy(), coeffs.detach().cpu().numpy()


def tv(img, weight):
    tv_h = ((img[1:, :] - img[:-1, :]).abs()).sum()
    tv_w = ((img[:, 1:] - img[:, :-1]).abs()).sum()
    return weight * (tv_h + tv_w)

def center_crop(measurement, des_shape):
    # Center crop 
    m_center = (measurement.shape[0]//2, measurement.shape[1]//2)
    left, right, up, down = ( m_center[1] - des_shape[1]//2, m_center[1] + int(np.round(des_shape[1]/2)),  \
                              m_center[0] - des_shape[0]//2, m_center[0] + int(np.round(des_shape[0]/2)))
    # TODO: Debug this for images of an odd size.
    measurement = measurement[left:right,up:down]
    return measurement


def wiener_torch(image, psf, balance=3e-4, reg=None, is_real=True, clip=True):
    if reg is None:
        reg, _ = laplacian(image.ndim, image.shape, is_real=is_real)
        reg = reg.to(device=image.device)

    trans_func = ir2tf(psf, image.shape, is_real=is_real).to(device=image.device)

    wiener_filter = torch.conj(trans_func) / (torch.abs(trans_func) ** 2 +
                                           balance * torch.abs(reg) ** 2)

    deconv = fft.irfftn(wiener_filter * fft.rfftn(image))

    if clip:
        deconv[deconv > 1] = 1
        deconv[deconv < -1] = -1

    return deconv


def laplacian(ndim, shape, is_real=True):
    impr = torch.zeros([3] * ndim)
    for dim in range(ndim):
        idx = tuple([slice(1, 2)] * dim +
                    [slice(None)] +
                    [slice(1, 2)] * (ndim - dim - 1))
        impr[idx] = torch.tensor([-1.0,
                              0.0,
                              -1.0]).reshape([-1 if i == dim else 1
                                              for i in range(ndim)])
    impr[(slice(1, 2), ) * ndim] = 2.0 * ndim
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
            irpadded = torch.roll(irpadded,
                               shifts=-int(np.floor(axis_size / 2)),
                               dims=axis)

    return fft.rfftn(irpadded, dim=list(range(-dim, 0)))


def image_quad_norm(image):
    np.sum(np.sum(np.abs(image) ** 2, axis=-1), axis=-1)