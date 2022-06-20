#code containing methods utilizing the linear revolution invariant:
#import polarTransform as pt
import numpy as np
import matplotlib.pyplot as plt
from _src import util, polar_transform, seidel
from tqdm import tqdm
import torch
import torch.fft as fft
import pdb


def blur(obj, psf_tensor_fft, method='normal', device=None, verbose=False, diff=None):

    # get appropriate radii for chosen PSF sampling
    num_radii = psf_tensor_fft.shape[0]
    dim = obj.shape[0]
    r_list = torch.linspace(0, np.sqrt(2)/2 * ((num_radii-1)/(dim//2)), num_radii)

    # get object RoFT
    obj_polar = polar_transform.img2polar(obj)
    obj_fft = fft.rfft(obj_polar, dim=0)

    # create blank image RoFT which will be subsequently filled in
    img_polar_fft = torch.zeros_like(obj_fft, dtype=torch.complex64, device=device)
    if method == 'alt':
        print('Using the alt (matrix based) method')
        freqs = tqdm(range(psf_tensor_fft.shape[1])) if verbose else (range(psf_tensor_fft.shape[1]))
        for i in freqs:
            psf_slice = psf_tensor_fft[:, i, :].T
            recon_row = torch.matmul(psf_slice, (obj_fft[i, :] * torch.FloatTensor(r_list))[:, None])
            img_polar_fft[i, :] = recon_row.T
    else:
        radii = tqdm(enumerate(r_list)) if verbose else (enumerate(r_list))
        for index, r in radii:
            curr_psf_polar_fft = psf_tensor_fft[index, :, :]
            img_polar_fft += (r*(obj_fft[:, index][:, None] * curr_psf_polar_fft))

    img_polar = fft.irfft(img_polar_fft, dim=0)
    img_polar = torch.roll(img_polar, int(np.ceil((45.0 / 360.0) * img_polar.shape[0])), dims=0)
    img = polar_transform.polar2img(img_polar, obj.shape)

    #return img
    img.data[img.data<0] = 0

    # calibrate?
    #pdb.set_trace()
    if diff is not None:
        pre_cal_mean = img.mean()
        #diff[img<1e-6] = 0
        #img = img - diff
        norm_diff = diff - diff.min()
        norm_diff = norm_diff/norm_diff.max()
        img = img + torch.mul(obj, norm_diff)*img.max()
        post_cal_mean = img.mean()
        img = img - (post_cal_mean - pre_cal_mean)
    #img = torch.tensor(im.shift(img.detach(), (0.5, 0.5)))
    #pdb.set_trace()
    return img

