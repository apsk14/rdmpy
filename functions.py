#Code implementing the user functions for lri-deblur. These functions can be directly imported via from lri-deblur import calibrate, blur, deblur
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import torch
import numpy as np
import torch.fft as fft
from _src import opt, seidel, forward, util
import pdb

'''
Takes in a calibration image with a few randomly placed PSFs OR Seidel coefficents, fits Seidel coefficients (if none provided) to the PSFS, and 
gives back the PSF stack needed for LRI from those Seidel coefficients.

Arguments:
    calib_image - (N x N) numpy array with randomly scattered PSFs

    num_radii - positive integer which determines how many radii are used for blurring/deblurring determines speed and memory size (recommended set to N).

    seidel_coeffs - (6 x 1) numpy array. The primary seidel coefficients with a defocus parameter appended before

    centered_psf - boolean (True or False). Specifies whether a PSF is centered in the image.
    
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
def calibrate(calib_image, model, dim, num_psfs=0, seidel_coeffs=None, get_psfs=True, fit_params={}, sys_params={}, verbose=False, show_psfs=False, device=torch.device('cpu')):

    # default parameters which describe the optical system. These do not necessarily need to be correct---they just need to generate accurate PSFs
    def_sys_params = {'samples': dim, 'L': 1e-3, 'lamb': 0.55e-6,
                      'pupil_radius': ((dim) * (0.55e-6) * (100e-3))/(4*(1e-3)), 'z': 100e-3}
    def_sys_params.update(sys_params)      

    # If no seidel coefficients were provided we will our own set to the calibration image
    if seidel_coeffs is None:

        # parameters which are used for the seidel fitting procedure
        def_fit_params = {'sys_center': [calib_image.shape[0]//2, calib_image.shape[1]//2], 'centered_psf': False, 'min_distance': 30, 'threshold': 0.2, 
                            'num_seidel': 3, 'init': 'zeros', 'seidel_init': None, 'iters': 300, 'lr': 1e-2, 'reg': 0, 'plot_loss': False}
        def_fit_params.update(fit_params)      

        # seperating out individual PSFs from the calibration image
        psf_locations, calib_image = util.get_calib_info(calib_image, dim, def_fit_params)

        # seidel fitting
        if verbose:
            print('fitting seidel coefficients...')
        seidel_coeffs = opt.estimate_coeffs(calib_image, psf_list=psf_locations, sys_params=def_sys_params, fit_params=def_fit_params, show_psfs=show_psfs, device=device).detach()
        if verbose:
            print('Fitted seidel coefficients: ' + str(seidel_coeffs.detach().cpu()))

    else:
        seidel_coeffs = torch.tensor(seidel_coeffs)

    
    if get_psfs:
        # Next, with seidel coefficients, we decide which PSFs we need depending on our method
        if model == 'lsi':
            point_list = [(0,0)] # just the center PSF
        elif model == 'lri':
            if num_psfs == 0:
                num_psfs = dim
            rs = np.linspace(0, (dim/2), num_psfs+1, endpoint=False, retstep=False)
            rs = rs[1:] - 0.5
            point_list = [(r , -r) for r in rs] # radial line of PSFs
        else:
            raise(NotImplementedError)

        if verbose:
                print('computing radial stack of PSFs...')
        psf_data = seidel.compute_psfs(seidel_coeffs.cpu(), point_list, def_sys_params, polar=(model=='lri'), verbose=verbose, stack=False)
        
        # prep the PSFs for outputing to the user
        if model == 'lsi':
            psf_data = psf_data[0].to(device)
        if model == 'lri':
            psf_data = torch.stack(psf_data)
            psf_data = fft.rfft(psf_data, dim=1).to(device) # here we store the PSF ROFT to save computation time later.

        return seidel_coeffs, psf_data

    else:
        
        return seidel_coeffs


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
def blur(obj, psf_data, model, verbose=False, device=torch.device('cpu')):
    
    obj = torch.from_numpy(obj).float().to(device)
    if model == 'lsi':
        if len(psf_data.shape) != 2: 
            raise ValueError('There is a mismatch between model and psf_data, LSI needs a single PSF') 
        img = forward.lsi(obj, psf_data)
    elif model == 'lri':
        if len(psf_data.shape) != 3:
            raise ValueError('There is a mismatch between model and psf_data, LRI needs a radial stack of PSFs')
        img,_ = forward.lri(obj, psf_data, method='normal', device=device, verbose=verbose)
    else:
        raise NotImplemented

    return img




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
def deblur(image, psf_data, model, opt_params={}, device=torch.device('cpu'), verbose=False):
    
    if model != 'lri' and model != 'lsi':
        raise NotImplementedError 
    
    if model == 'lsi' and len(psf_data.shape) != 2: 
        raise ValueError('There is a mismatch between model and psf_data, LSI needs a single PSF') 

    if model == 'lri' and len(psf_data.shape) != 3:
        raise ValueError('There is a mismatch between model and psf_data, LRI needs a radial stack of PSFs')

    def_opt_params = {'iters': 100, 'optimizer': 'adam', 'lr': 7.5e-2, 'init': 'measurement', 'crop': 0, 'tv_reg': 1e-11, 'l2_reg': 0, 'plot_loss': False}
    def_opt_params.update(opt_params)
    
    if not torch.is_tensor(image):
        image = torch.tensor(image)
    if image.device is not device:
        image = image.to(device)


    return opt.image_recon(image.float(), psf_data, model, opt_params=def_opt_params, device=device, verbose=verbose)


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
def blind_deblur(image, opt_params={}, sys_params={}, verbose=False, device=torch.device('cpu')):
    def_opt_params = {'iters': 100, 'optimizer': 'adam', 'lr': 7.5e-2, 'init': 'measurement', 'seidel_init': None,  'crop': 0, 'tv_reg': 0, 'l2_reg': 0, 'balance': 3e-4, 'plot_loss': False}
    def_opt_params.update(opt_params)

    def_sys_params = {'samples': min(image.shape), 'L': 1e-3, 'lamb': 0.55e-6,
                      'pupil_radius': ((min(image.shape)) * (0.55e-6) * (100e-3))/(4*(1e-3)), 'z': 100e-3}
    def_sys_params.update(sys_params)  

    if not torch.is_tensor(image):
        image = torch.tensor(image).float()
    if image.device is not device:
        image = image.to(device)

    return opt.blind_recon(image, opt_params=def_opt_params, sys_params=def_sys_params, device=device)