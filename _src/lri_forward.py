#code containing methods utilizing the linear revolution invariant:
#import polarTransform as pt
import numpy as np
from . import  polar_transform, util, seidel
from tqdm import tqdm
import torch
import torch.fft as fft
import pdb
from scipy.ndimage import shift


def forward(obj, psf_tensor_fft, method='normal', device=None, verbose=False, diff=None, calib=False):

    # get appropriate radii for chosen PSF sampling
    #L = 7.33333333333 * 0.0003072
    center_psf = psf_tensor_fft[0]
    psf_tensor_fft = psf_tensor_fft[1]
    #psf_tensor_fft = psf_tensor_fft.to(torch.float64)
    num_radii = psf_tensor_fft.shape[0]
    dim = obj.shape[0]
    #r_list = torch.linspace(0, np.sqrt(2)/2 * ((num_radii-1)/(dim//2)), num_radii)
    #r_list = np.linspace(0, np.sqrt(2)/2 * ((num_radii-1)/(dim//2)), num_radii, endpoint=True)#(np.sqrt(2)*(np.linspace(0, dim/2, num_radii, endpoint=False) + 0.5))/(dim/2)
    r_list = np.linspace(np.sqrt(2)/2, dim/(np.sqrt(2)), num_radii)

    # get object RoFT
    obj_polar = polar_transform.img2polar(obj, numRadii=num_radii)
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
            curr_psf_polar_fft = curr_psf_polar_fft/curr_psf_polar_fft.sum()
            img_polar_fft += ((obj_fft[:, index][:, None] * curr_psf_polar_fft))
            #img_polar_fft[:, index] = ((obj_fft[:, index] * curr_psf_polar_fft[:, index]))

    img_polar = fft.irfft(img_polar_fft, dim=0)
    #img_polar = torch.roll(img_polar, int(np.ceil(-(45.0 / 360.0) * img_polar.shape[0])), dims=0)
    #img_polar = torch.tensor(shift(img_polar.cpu(), (-(45.0 / 360.0) * img_polar.shape[0], 0),  mode='wrap',order=3  )).to(device)


    img = polar_transform.polar2img(img_polar, obj.shape)
    #pdb.set_trace()
    #img = img + obj[dim//2,dim//2]*(center_psf *(img.max()/center_psf.max()))
    img.data[img.data<0] = 0
    #img = img*(obj.max()/img.max())


    
    # (obj.sum() -  obj[dim//2,dim//2])


    #  add in the center psf (missing from the 0 radius point)
    # if not calib:
    #     img = img + obj[dim//2,dim//2]*center_psf 
    
    # polar sampling artifact correction
    if (diff is not None):
        pre_cal_mean = img.mean()
        norm_diff = diff - diff.min()
        norm_diff = norm_diff/norm_diff.max()
        img = img + torch.mul(obj, norm_diff)*img.max()
        post_cal_mean = img.mean()
        img = img - (post_cal_mean - pre_cal_mean)

    # scale adjustment

    img = img/(img.sum())
    img = img * obj.sum()

    #center = dim//2 
    #img[center - 10: center+10, center - 10: center+10 ] = 0
    # img = img/(img.sum())
    # img = (img * (obj.sum()-obj[center,center]))# - obj[dim//2,dim//2])) + obj[dim//2,dim//2]*(center_psf)
    # img = img + center_psf*obj[center, center]


    return img, obj_polar

