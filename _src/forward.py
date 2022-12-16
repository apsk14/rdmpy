#code containing methods utilizing the linear revolution invariant:
#import polarTransform as pt
import numpy as np
from torch_fftconv import fft_conv2d
from . import  util, seidel, polar_transform
#from . import pt_old as polar_transform
from tqdm import tqdm
import torch
import torch.fft as fft
import pdb
from scipy.ndimage import shift
import matplotlib.pyplot as plt


def lri(obj, psf_tensor_fft, method='normal', device=torch.device('cpu'), verbose=False, diff=None, calib=False):

    # infer info from the PSF roft
    num_radii = psf_tensor_fft.shape[0]
    dim = obj.shape[0]

    r_list = np.linspace(0, (dim/2) * np.sqrt(2), num_radii+1, endpoint=False)
    r_list = r_list[1:]
    #r_list = np.linspace(0, (dim/2) * np.sqrt(2), num_radii, endpoint=False)
    # r_list = r_list[1:]

    # get object RoFT
    obj_polar = polar_transform.img2polar(obj, numRadii=num_radii)
    obj_fft = fft.rfft(obj_polar, dim=0)

    # create blank image RoFT which will be subsequently filled in
    img_polar_fft = torch.zeros_like(obj_fft, dtype=torch.complex64, device=device)
    if method == 'alt':
        r_list = torch.tensor(r_list, device=device).float()
        print('Using the alt (matrix based) method')
        freqs = tqdm(range(psf_tensor_fft.shape[1])) if verbose else (range(psf_tensor_fft.shape[1]))
        for i in freqs:
            psf_slice = psf_tensor_fft[:, i, :].T
            recon_row = torch.matmul(psf_slice, (obj_fft[i, :] * r_list)[:, None])
            img_polar_fft[i, :] = recon_row.T
    else:
        radii = tqdm(enumerate(r_list)) if verbose else (enumerate(r_list))
        for index, r in radii:
            curr_psf_polar_fft = psf_tensor_fft[index, :, :]
            #curr_psf_polar_fft = curr_psf_polar_fft/curr_psf_polar_fft.abs().sum()
            img_polar_fft += (r*(obj_fft[:, index][:, None] * curr_psf_polar_fft))

    img_polar = fft.irfft(img_polar_fft, dim=0)
    img = polar_transform.polar2img(img_polar, obj.shape)

    #img = img*(obj.max()/img.max())
    #img.data[img.data<0] = 0
    
    # polar sampling artifact correction
    # if (diff is not None):
    #     pre_cal_mean = img.mean()
    #     norm_diff = diff - diff.min()
    #     norm_diff = norm_diff/norm_diff.max()
    #     img = img + torch.mul(obj, norm_diff)*img.max()
    #     post_cal_mean = img.mean()
    #     img = img - (post_cal_mean - pre_cal_mean)

    # scale adjustment
    img = img/(img.sum())
    img = img * obj.sum()

    return img, obj_polar


def lsi(obj, psf):
    img = fft.irfftn(fft.rfftn(obj) * fft.rfftn(psf))
    return torch.roll(img, shifts=(-img.shape[0] // 2, -img.shape[1] // 2), dims=(0, 1))