#utility methods
import numpy as np
import os, shutil
import pdb
import scipy.sparse as sp
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.image as py
import math
import scipy.ndimage as im
from skimage.transform import resize
#import cv2
from cv2 import VideoWriter, VideoWriter_fourcc
from skimage.feature import corner_peaks
from skimage.morphology import (erosion, disk)
import torch
PLT_SIZE = 5

#TODO: get diagonal list given non-square dimensions

def write_video(path, array, fps=float(24)):
    fourcc = VideoWriter_fourcc(*'mp4v')
    video = VideoWriter(path, fourcc, fps, (array.shape[2], array.shape[1]), False)
    for i in range(array.shape[0]):
        video.write( (array[i, :, :]*255).astype(np.uint8))
    video.release()


def getSeidelList(sidelength, num_radii):
    fraction = (sidelength/2)/num_radii
    diag_list = []
    for i in range(0, num_radii): #start was 0!
        diag_list += [((i*fraction),  (i*fraction))]
    return diag_list


def get_calib_info(calib_image, center, desired_dim, centered_psf=True):
    psf = calib_image.copy()
    psf[psf < 0] = 0
    psf[psf < np.quantile(psf, 0.9)] = 0
    raw_coord = corner_peaks(erosion(psf, disk(2)), min_distance=30, indices=True, threshold_rel=0)
    distances = np.sqrt(np.sum(np.square(raw_coord - center), axis=1))
    if centered_psf:
        center = raw_coord[np.argmin(distances), :]

    calib_image = calib_image[center[0] - desired_dim[0] // 2:center[0] + desired_dim[0] // 2,
          center[1] - desired_dim[1] // 2:center[1] + desired_dim[1] // 2]

    coord_list = []
    for i in range(raw_coord.shape[0]):
        if np.abs(raw_coord[i, 1] - center[1]) < desired_dim[1] // 2 and np.abs(center[0] - raw_coord[i, 0]) < desired_dim[0] // 2:
            coord_list += [(raw_coord[i, 1] - center[1], center[0] - raw_coord[i, 0])]

    calib_image[calib_image < 0] = 0
    calib_image[calib_image < np.quantile(calib_image, 0.9)] = 0
    calib_image = (calib_image / calib_image.sum()) * len(coord_list)

    return coord_list, center, calib_image



def align_psf(psf, center_coords, output_shape): #in (x,y)
    if psf.shape[0] > output_shape[0]:
        amt = (psf.shape[0] - output_shape[0])//2
        psf = psf[amt:-amt, amt:-amt]
    else:
        psf = np.pad(psf, ((output_shape[0] - psf.shape[0])//2, (output_shape[1] - psf.shape[1])//2), 'constant',
                     constant_values=(0, 0)) #fit onto sensor
    pixel_shift_x = center_coords[0]
    pixel_shift_y = center_coords[1]
    if pixel_shift_x > 0.0:
        psf = im.shift(psf,(0,pixel_shift_x))
        #psf[:,0:int(np.ceil(pixel_shift_x))] = 0
    elif pixel_shift_x < 0.0:
        psf = im.shift(psf,(0,pixel_shift_x))
        #psf[:,int(np.ceil(pixel_shift_x)):] = 0
    if pixel_shift_y > 0.0:
        psf = im.shift(psf,(pixel_shift_y,0))
        #psf[0:int(np.ceil(pixel_shift_y)),:] = 0
    elif pixel_shift_y < 0.0:
        psf = im.shift(psf,(pixel_shift_y,0))
        #psf[int(np.ceil(pixel_shift_y)):,:] = 0

    psf[psf < 0] = 0 #make sure no negative values came in from interpolation
    return psf


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def rm(path, dir=False):
    if os.path.exists(path):
        if dir:
            shutil.rmtree(path)
        else:
            os.remove(path)


def resize_complex(mat, shape):
    reals = np.real(mat)
    imag = np.imag(mat)
    return resize(reals, shape) + resize(imag, shape)


def save(path, obj, type):
    if type == 'npy':
        np.save(path, obj)
    elif type == 'npz':
        sp.save_npz(path, obj)
    elif type == 'pkl':
        pkl.dump(obj, open(path, 'wb'))
    else:
        print('please enter in a valid type')
        exit()
    return

def reshape_fortran(x, shape):
    if len(x.shape) > 0:
        x = x.permute(*reversed(range(len(x.shape))))
    return x.reshape(*reversed(shape)).permute(*reversed(range(len(shape))))

def plot_mult(I, cmap='gray', path='', show=True):
    plt.figure()

    #  set plot size
    fig = plt.gcf()
    fig.set_size_inches(len(I)*PLT_SIZE, PLT_SIZE)

    #  code for making image and colorbar the same length
    ax = plt.gca()
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    im = ax.imshow(np.concatenate(I, axis=1), cmap=cmap)
    divider = make_axes_locatable(ax)
    cax = divider.new_vertical(size="5%", pad=0.05, pack_start=True)
    fig.add_axes(cax)
    fig.colorbar(im, cax=cax, orientation='horizontal') #orientation='horizontal',

    #  show and/or saveout
    if show: plt.show()
    if path != '':
        plt.savefig(path, orientation='landscape', bbox_inches='tight')

def show(I):
    plt.figure()
    plt.imshow(I, cmap='gray')
    plt.show()

def error_map(img, gt, path, display=False, cmap='magma'):
    error = np.abs(img - gt)
    plot_mult([error], path=path, show=display, cmap=cmap)
    # plt.figure()
    # plt.imshow(error, cmap=cmap)
    # plt.colorbar()
    # if display:
    #     plt.show()
    # if path != '':
    #     plt.savefig(path)
    #     #py.imsave(path, error, cmap=cmap)
    return error


def mse(im1, im2):
    return ((im1-im2)**2).mean()

def normalize(I):
    I = I - I.min()
    I = I / I.max()
    return I

def add_noise(obj, noise):
    if noise.type == 'gaussian':
        mean = noise.params[0]
        std = noise.params[1]**0.5
        obj_noisy = obj + noise.amt*np.random.normal(mean, std, obj.shape)
    else:
        raise NotImplemented
    obj_noisy[obj_noisy < 0] = 1
    obj_noisy[obj_noisy > 1] = 1
    return obj_noisy


def list_diff(list1, list2):
    return [x for x in list1 if x not in list2]

def coord_to_radii(coords):
    return [np.sqrt(coord[0]**2 + coord[1]**2) for coord in coords]

def process_video(stack, back, center, dim):
    new_stack = np.zeros((stack.shape[0], dim[0], dim[1]))
    for i in range(stack.shape[0]):
        new_stack[i, :, :] = process(stack[i, :, :], back, center, dim)
    return new_stack

def process(test, back, center, dim):
    test = test - back
    test[test < 0] = 0
    test_image = test[center[0] - dim[0] // 2:center[0] + dim[0] // 2,
    center[1] - dim[1] // 2:center[1] + dim[1] // 2]

    test_image = test_image - test_image.min()
    test_image = (test_image/test_image.max())*0.90

    return test_image


def center_crop(measurement, des_shape):
    # Center crop
    m_center = (measurement.shape[0]//2, measurement.shape[1]//2)
    left, right, up, down = ( m_center[1] - des_shape[1]//2, m_center[1] + int(np.round(des_shape[1]/2)),  \
                              m_center[0] - des_shape[0]//2, m_center[0] + int(np.round(des_shape[0]/2)))
    # TODO: Debug this for images of an odd size.
    measurement = measurement[left:right,up:down]
    return measurement