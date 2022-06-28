#Code implementing the user functions for lri-deblur. These functions can be directly imported via from lri-deblur import calibrate, blur, deblur

import torch
import numpy as np
import torch.fft as fft
from ._src import opt, seidel, lri, util
import pdb

'''
Takes in a calibration image with a few randomly placed PSFs OR Seidel coefficents, fits Seidel coefficients (if none provided) to the PSFS, and 
gives back the PSF stack needed for LRI from those Seidel coefficients.

Arguments:
    calib_image - (N x N) numpy array with randomly scattered PSFs

    num_radii - positive integer which determines how many radii are used for blurring/deblurring determines speed and memory size (recommended set to N).

    seidel_coeffs - (6 x 1) numpy array. The primary seidel coefficients with a defocus parameter appended before

    sys_params -  parameters of the system takes the form of a dictionary: {'samples': N, 'L': sidelength length of sensor, 'lamb': wavelength,
                      'pupil_radius': radius of system pupil, 'z': distance from pupil to sensor}. All lengths are in millimeters

    opt_params - optimization parameters used for the Seidel fitting procedure. Takes the form of a dictionary: {'iters': number of iterations, 'lr': learning rate, 'reg': l2 regularization}

    verbose - Boolean (True or False). Determines if the code will output descriptive print statements.

    device - torch.device. Determines the device all quantities will be placed on. Will be cuda or cpu depending on if a GPU is being used.

Returns: 
    psf_stack_roft - (num_radii x N x N) torch tensor. Can be viewed as a num_radii stack of N x N images where each image is the rotationaly fourier transform of a PSF. The PSFs are ordered 
    by increasing distance of their generating point source from the center.

    seidel_coeffs - (6 x 1) numpy array storing the 5 primary coefficients preceded by a defocus parameter. These coefficients were used to generate psf_stack_roft
'''
def calibrate(calib_image, desired_dim, num_psfs, seidel_coeffs=None, opt_params=None, sys_params=None,  verbose=True, device=torch.device('cpu')):
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
        psf_locations, center, calib_image = util.get_calib_info(calib_image, center, [desired_dim, desired_dim])  # going to be in (x,y) relative to center. Center will be row, col from top left of raw_image

        '''Need to generalize this to multishot'''
        if verbose:
            print('estimating seidel coefficients...')
        seidel_coeffs = opt.estimate_coeffs_ss(calib_image, psf_locations, sys_params, opt_params, device)

    # Next, with seidel coefficients, we generate the radial line of PSFs needed for LRI operations
    point_list = util.getSeidelList(desired_dim, num_psfs)

    if verbose:
            print('computing radial stack of PSFs...')
    psf_stack = seidel.compute_psfs(seidel_coeffs, point_list, sys_params, device=device, polar=True, verbose=verbose)
    psf_stack = torch.stack(psf_stack)
    psf_stack_roft = fft.rfft(psf_stack, dim=1).to(device)

    return psf_stack_roft, seidel_coeffs



'''
Takes in an object/scene and radial stack of PSF rotationally fourier transforms, and returns a blurred image from the system with the PSFs in the stack. 
Computes the LRI forward model to get the blurred image.

Arguments:
    obj - (N x N) numpy array representing an object that the user would like to blur. 

    psf_stack_roft - (num_radii x N x N) torch tensor. Can be viewed as a num_radii stack of N x N images where each image is the rotationaly fourier transform of a PSF. The PSFs are ordered 
    by increasing distance of their generating point source from the center. Can be gotten from calibrate.

    verbose - Boolean (True or False). Determines if the code will output descriptive print statements.

    artifact_correction - real number in (0,1). Mitigates image artifacts caused by discretizing the polar transformation. Costs a single
    run of blur function. Works by subtracting the output of a constant object. Its value determines the strength of correction.

    device - torch.device. Determines the device all quantities will be placed on. Will be cuda or cpu depending on if a GPU is being used.

Returns
    blurred_image - (N x N) numpy array that is a blurred version of the object according to the PSFs from psf_stack_roft
'''
def blur(obj, psf_stack_roft, verbose=False, artifact_correction=0.5, device=torch.device('cpu')):
    #diff image stuff
    if artifact_correction:
        constant = torch.ones(obj.shape, device=device)  * artifact_correction
        constant_response = lri.blur(constant, psf_stack_roft, method='normal', device=device, verbose=verbose)
        diff = constant - constant_response
    else:
        diff = None

    obj = torch.from_numpy(obj).float().to(device)
    return lri.blur(obj, psf_stack_roft, method='normal', device=device, verbose=verbose, diff=diff)




'''
Takes in a blurry image and the radial stack of PSF rotationally fourier transforms from the same system. Inverts the LRI forward model
iteratively to return an estimation of the unblurred object/scene.

Arguments:
    image - (N x N) numpy array representing a blurry image that we would like to deblur

    psf_stack_roft - (num_radii x N x N) torch tensor. Can be viewed as a num_radii stack of N x N images where each image is the rotationaly fourier transform of a PSF. The PSFs are ordered 
    by increasing distance of their generating point source from the center. Can be gotten from calibrate.

    verbose - Boolean (True or False). Determines if the code will output descriptive print statements.

    artifact_correction - real number in (0,1). Mitigates image artifacts caused by discretizing the polar transformation. Costs a single
    run of blur function. Works by subtracting the output of a constant object. Its value determines the strength of correction.

    device - torch.device. Determines the device all quantities will be placed on. Will be cuda or cpu depending on if a GPU is being used.

Returns
    blurred_image - (N x N) numpy array that is a blurred version of the object according to the PSFs from psf_stack_roft
'''
def deblur(image, psf_stack_roft, opt_params=None, verbose=False, artifact_correction=0.5, device=torch.device('cpu')):
    if artifact_correction:
        constant = torch.ones(image.shape, device=device)  * artifact_correction
        constant_response = lri.blur(constant, psf_stack_roft, method='normal', device=device, verbose=verbose)
        diff = constant - constant_response
    else:
        diff = None

    if opt_params is None:
        opt_params = {'iters': 100, 'optimizer': 'adam', 'lr': 7.5e-2, 'init': 'measurement', 'crop': 0, 'reg': 1e-11}
    if not torch.is_tensor(image):
        image = torch.tensor(image, device=device).float()
    return opt.image_recon(image, psf_stack_roft, opt_params, diff, device)