#used for functions which estimate PSFs given a set of PSFs
"""
Amit Kohli
3-23-21
This file contains functions for computing PSFs from seidel coefficients and for estimating seidel
coefficients from PSF data
"""

from ast import Num
import csv
from tqdm import tqdm
import scipy as sp
from _src import util, polar_transform
from bisect import bisect
from scipy.ndimage import shift
import numpy as np
import torch
import torch.nn.functional as fun
from torch.fft import fftn, ifftn
import matplotlib.pyplot as plt
import pathlib
from skimage.restoration import unwrap_phase as unwrap
dirname = str(pathlib.Path(__file__).parent.parent.absolute())
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 500
import pdb

# Note coeffs expected in following order: Wd, W040, W131, W222, W220, W311

def compute_pupil_phase(coeffs, X, Y, u, v):
    #  first rotate grid as needed
    rot_angle = torch.atan2(v, u)
    obj_rad = torch.sqrt(u**2 + v**2)
    X_rot = X*torch.cos(rot_angle) + Y*torch.sin(rot_angle)
    Y_rot = -X*torch.sin(rot_angle) + Y*torch.cos(rot_angle)

    pupil_radii = torch.square(X_rot) + torch.square(Y_rot)
    pupil_phase = coeffs[0]*pupil_radii + coeffs[1]*torch.square(pupil_radii) \
                 + coeffs[2]*obj_rad*pupil_radii*X_rot + coeffs[3]*(obj_rad**2)*torch.square(X_rot) \
                 + coeffs[4]*(obj_rad**2)*pupil_radii + coeffs[5]*(obj_rad**3)*X_rot

    return pupil_phase

def compute_psfs(coeffs, desired_list, sys_params, device=torch.device('cpu'), polar=False, verbose=False, stack=False):
    num_radii = len(desired_list)
    desired_list = [(torch.tensor(i[0], device=device).float(), torch.tensor(i[1], device=device).float()) for i in desired_list]
    iterable_coords = tqdm(desired_list) if verbose else desired_list
    samples = sys_params['samples']
    L = sys_params['L']
    dt = L / samples
    lamb = sys_params['lamb']
    k = (2 * np.pi) / lamb
    pupil_radius = sys_params['pupil_radius']
    z = sys_params['z']
    # f_number = z/(2*pupil_radius)
    # cutoff_freq = 1/(lamb * f_number
    fx = np.linspace(-1 / (2 * dt), 1 / (2 * dt), samples)
    [Fx, Fy] = torch.tensor(np.meshgrid(fx, fx), device=device)
    scale_factor = ((lamb * z) / pupil_radius)
    circle = circ(torch.sqrt(torch.square(Fx) + torch.square(Fy)) * scale_factor, radius=1)
    if stack:
        if polar: 
            if samples > 500:
                desired_psfs = torch.zeros((samples, samples*2, samples), device=device)
            else:
                desired_psfs = torch.zeros((samples, samples*4, samples), device=device)
        else:
            desired_psfs = torch.zeros((samples, samples, samples), device=device)
    else:
        desired_psfs = []
    idx = 0
    for point in iterable_coords:
        W = compute_pupil_phase(lamb*coeffs, X=-Fx * scale_factor, Y=-Fy * scale_factor, u=(point[0])/(samples/2), v=-(point[1])/(samples/2)) #point[0]/(sys_params['samples']/2)
        H = circle * torch.exp(-1j * k * W)  # weird phase issue here?"
        H[circle < 1e-12] = 0
        coherent_psf = torch.fft.ifftn(H)
        curr_psf = torch.roll(coherent_psf, shifts=(-coherent_psf.shape[0] // 2, -coherent_psf.shape[1] // 2),
                              dims=(0, 1))
        curr_psf = torch.square(torch.abs(curr_psf))
        curr_psf = fun.pad(curr_psf, (int(samples / 2), int(samples / 2), int(samples / 2), int(samples / 2)))
        curr_psf = torch.roll(curr_psf, (-int(point[1]), int(point[0])), dims=(0, 1))
        curr_psf = curr_psf[int(samples / 2):-int(samples / 2), int(samples / 2):-int(samples / 2)]
        curr_psf = curr_psf / curr_psf.sum()
        if polar:
            curr_psf = polar_transform.img2polar(curr_psf.float(), numRadii=num_radii)
        if stack:
            desired_psfs[idx, :, :] = curr_psf.to(device)
        else:
            desired_psfs += [curr_psf.to(device)]
        idx += 1


    return desired_psfs


def generate_psf_dataset(path, sys_params, coeffs, type='grid'):


    coeffs = torch.tensor(coeffs).float()
    samples = sys_params['samples']
    dt = sys_params['L']/sys_params['samples']
    scale_factor = ((sys_params['lamb'] * sys_params['z']) / sys_params['pupil_radius'])
    fx = np.linspace(-1 / (2 * dt), 1 / (2 * dt), samples, endpoint=False)
    [Fx, Fy] = torch.tensor(np.meshgrid(fx, fx))  # torch.meshgrid ordered ij not xy so we use numpy instead
    k = (2 * np.pi) / sys_params['lamb']

    circle = circ(torch.sqrt(torch.square(Fx) + torch.square(Fy)) * scale_factor, radius=1)

    util.mkdir(path)
    f = open(path + 'sys_params.txt', 'w')
    f.write(str(sys_params))
    f.close()

    print(coeffs)
    np.save(path + 'coeffs.npy', coeffs.detach().numpy())

    data_dict = {}
    index = 1
    if type == 'grid':
        u_norm = torch.linspace(-1, 1, samples)
        v_norm = -1 * torch.linspace(-1, 1, samples)
        for row, v in tqdm(zip(range(samples), v_norm)):
            for col, u in zip(range(samples), u_norm):
                W = compute_pupil_phase(coeffs * sys_params['lamb'], X=-Fx*scale_factor, Y=-Fy*scale_factor, u=u, v=v)
                H = circle * torch.exp(-1j * k * W)
                H[circle < 1e-12] = 0
                curr_psf = torch.fft.ifft(torch.fft.ifft(H, dim=0), dim=1) #torch.fft.ifftn(H,) WTF IS THIS BUG POWERS OF 2 ARE BROKEN?!!!
                #curr_psf = torch.fft.ifftn(H)
                curr_psf = torch.roll(curr_psf, shifts=(curr_psf.shape[0] // 2, curr_psf.shape[1] // 2), dims=(0, 1))
                curr_psf = torch.square(torch.abs(curr_psf))
                curr_psf = fun.pad(curr_psf, (int(samples/2), int(samples/2), int(samples/2), int(samples/2)))
                curr_psf = torch.roll(curr_psf, (-int(row - samples/2), int(col - samples/2)), dims=(0, 1))
                curr_psf = curr_psf[int(samples/2):-int(samples/2), int(samples/2):-int(samples/2)]
                curr_psf = curr_psf/curr_psf.sum()
                curr_psf_sparse = sp.sparse.csc_matrix(curr_psf.detach().numpy())
                sp.sparse.save_npz(path+str(index), curr_psf_sparse)
                data_dict[(col, row)] = index
                index += 1
    elif type == 'line':
        u_norm = torch.linspace(0, 1, samples)
        v_norm = -1 * torch.linspace(0, 1, samples)
        for index, (u,v) in enumerate(zip(u_norm, v_norm)):
            W = compute_pupil_phase(coeffs * sys_params['lamb'], X=-Fx * scale_factor,
                                    Y=-Fy * scale_factor, u=u, v=v)
            H = circle * torch.exp(-1j * k * W)
            H[circle < 1e-12] = 0
            #coherent_psf = torch.fft.ifft(torch.fft.ifft(H, dim=0), dim=1)  # torch.fft.ifftn(H,) WTF IS THIS BUG POWERS OF 2 ARE BROKEN?!!!
            coherent_psf = torch.fft.ifftn(H)
            curr_psf = torch.roll(coherent_psf, shifts=(-coherent_psf.shape[0] // 2, -coherent_psf.shape[1] // 2),
                                  dims=(0, 1))
            curr_psf = torch.square(torch.abs(curr_psf))
            curr_psf = fun.pad(curr_psf, (int(samples / 2), int(samples / 2), int(samples / 2), int(samples / 2)))
            #curr_psf = torch.roll(curr_psf, (-int(index - samples / 2), int(index - samples / 2)), dims=(0, 1))
            curr_psf = sp.ndimage.shift(curr_psf, (-index/2, index/2), prefilter=False)
            curr_psf = curr_psf[int(samples / 2):-int(samples / 2), int(samples / 2):-int(samples / 2)]
            curr_psf = curr_psf / curr_psf.sum()
            curr_psf_sparse = sp.sparse.csc_matrix(curr_psf)
            sp.sparse.save_npz(path + str(index+1), curr_psf_sparse)
            data_dict[(index, index)] = index+1
    else:
        raise NotImplemented


    with open(path + 'data.csv', 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in data_dict.items():
            writer.writerow([key, value])



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
        psfs_estimate, pupils = compute_psfs(torch.cat((torch.zeros(1, 1, device=device), coeffs, torch.zeros(2, 1, device=device) )), desired_list=psf_list, sys_params=sys_params, device=coeffs.device)\
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



def circ(r, radius):
    return(abs(r)<=radius)

def main():

    ''' testing compute_pupil_phase'''
    # u = 1; v = 0
    # coeffs = [0, 1, 1, 1, 0, 0]
    # x = np.linspace(-1,1,1000)
    # y = np.linspace(-1,1,1000)
    # X, Y = np.meshgrid(x,y)
    # circle = circ(np.sqrt(np.square(X) + np.square(Y)), radius=1)
    # mask = circle==0
    # W = compute_pupil_phase(torch.tensor(coeffs), torch.tensor(X),
    #                         torch.tensor(Y), torch.tensor(u), torch.tensor(v)).detach().numpy()
    # W[mask] = np.nan
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # ax.plot_surface(X,Y,W)
    # plt.show()


    '''testing compute_psfs'''
    lamb = 0.55e-6
    sys_params = {'samples': 1024, 'L': 1e-3, 'lamb': lamb, 'pupil_radius': 10e-3, 'z': 100e-3}
    coeffs = [ 0, 4.963*lamb,	   2.637*lamb,	    9.025*lamb,	    7.536*lamb,	 0.157*lamb]
    # lamb = 0.510e-6
    # sys_params = {'samples': 384, 'L': 5.8*0.1326e-3, 'lamb': lamb, 'pupil_radius': 1e-3, 'z': 12.957e-3}
    #coeffs = [ 0,  0.05*lamb, -0.171021*lamb, 0.146097*lamb, 0*lamb, 0.009018*lamb]
    #coeffs = [0, 0.4* lamb, 0.4637 * lamb, 0.455 * lamb, 0 * lamb, 0.557 * lamb]

    compute_psfs(coeffs, desired_list=[(0,0), (0.5,0.5), (1,1)], sys_params=sys_params)

    '''testing estimate_coeffs'''
   #  lamb = 0.510e-6
   #  sys_params = {'samples': 384, 'L': 5.8*0.1326e-3, 'lamb': lamb, 'pupil_radius': 1e-3, 'z': 12.957e-3}
   #  opt_params = {'iters': 2000, 'lr': 1e-3, 'reg': 0}
   #  model_name = 'mscope2d_orig_12-04-20'
   #  cache_path = dirname + '/data/.cache/'
   #  psf_path = dirname + '/data/' + model_name + '/'
   #
   #  # Instantiate dataset object which will contain all the PSFs needed.
   #  dataset = PSFDataset.Dataset(name='model_name', psf_path=psf_path, small=True,
   #                               psf_shape=(384, 384), cache_path=cache_path)
   #  psf_list = [(0,0)]
   #  psfs = dataset.get_psfs(psf_list)
   # # psfs = [torch.load(cache_path+'psf_fit.pt')]
   #
   #  estimate_coeffs(psf_data=(psf_list, psfs), sys_params=sys_params, opt_params=opt_params)


if __name__ == '__main__':
    main()