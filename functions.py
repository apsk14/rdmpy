#Code implementing the user functions for lri-deblur. These functions can be directly imported via from lri-deblur import calibrate, blur, deblur

import torch
import numpy as np
import torch.fft as fft
from _src import opt, seidel, lri, util
import pdb


def calibrate(calib_image, desired_dim, num_radii, seidel_coeffs=None, sys_params=None, opt_params=None, verbose=True, device=torch.device('cpu')):
    # First step is to get the seidel coefficients by processing and fitting the calibration image
    if sys_params is None:
        sys_params = {'samples': desired_dim, 'L': 7.33333333333 * 0.0003072, 'lamb': 0.510e-6,
                      'pupil_radius': 1e-3, 'z': 12.957e-3}
    if seidel_coeffs is not None:
        seidel_coeffs = torch.tensor(seidel_coeffs)
    else:
        # default parameters
        if opt_params is None:
            opt_params = {'iters': 1000, 'lr': 2e-3, 'reg': 0}

        center = [calib_image.shape[0]//2, calib_image.shape[1]//2]
        center = (967, 1160)
        psf_locations, center, calib_image = util.get_calib_info(calib_image, center, [desired_dim, desired_dim])  # going to be in (x,y) relative to center. Center will be row, col from top left of raw_image
        print(center)

        '''Need to generalize this to multishot'''
        seidel_coeffs = opt.estimate_coeffs_ss(calib_image, psf_locations, sys_params, opt_params, device)

    # Next, with seidel coefficients, we generate the radial line of PSFs needed for LRI operations
    point_list = util.getSeidelList(desired_dim, num_radii)
    psf_stack = seidel.compute_psfs(seidel_coeffs, point_list, sys_params, device=device, polar=True, verbose=verbose, stack=True)
    psf_stack_roft = fft.rfft(psf_stack, dim=1).to(device)

    return psf_stack_roft, seidel_coeffs


def blur(obj, psf_stack_roft, verbose=False, artifact_correction=True, device=torch.device('cpu')):
    #diff image stuff
    if artifact_correction:
        if verbose: 'artifact suppression...'
        constant = torch.ones(obj.shape, device=device)  * 0.5
        constant_response = lri.blur(constant, psf_stack_roft, method='normal', device=device, verbose=verbose)
        diff = constant - constant_response
    else:
        diff = None

    if verbose: 'blurring...'
    obj = torch.from_numpy(obj).float().to(device)
    return lri.blur(obj, psf_stack_roft, method='normal', device=device, verbose=verbose, diff=diff)


def deblur(image, psf_stack_roft, opt_params=None, verbose=False, artifact_correction=True, device=torch.device('cpu')):
    if artifact_correction:
        if verbose: 'artifact suppression...'
        constant = torch.ones(image.shape, device=device)  * 0.5
        constant_response = lri.blur(constant, psf_stack_roft, method='normal', device=device, verbose=verbose)
        diff = constant - constant_response
    else:
        diff = None

    if verbose: 'deblurring...'
    if opt_params is None:
        opt_params = {'iters': 300, 'optimizer': 'adam', 'lr': 7.5e-2, 'init': 'measurement', 'crop': 0, 'reg': 1e-10}
    if not torch.is_tensor(image):
        image = torch.tensor(image, device=device).float()
    return opt.image_recon(image, psf_stack_roft, opt_params, diff, device)