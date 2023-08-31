"""
Utility methods shared by all files
"""
import os
import shutil

import math
import numpy as np
import torch
from torch.nn.functional import grid_sample
from scipy.ndimage import median_filter
from skimage.feature import corner_peaks
from skimage.morphology import erosion, disk

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from cv2 import VideoWriter, VideoWriter_fourcc

PLT_SIZE = 5


def write_video(path, array, fps=float(24)):
    """
    Writes a video from stack of images

    Parameters
    ----------
    path : str
        Path to save video to.

    array : np.ndarray
        Stack of frames to convert to video.

    fps : float, optional
        Frames per second of video.

    Returns
    -------
    None
    """

    fourcc = VideoWriter_fourcc(*"mp4v")
    video = VideoWriter(path, fourcc, fps, (array.shape[2], array.shape[1]), False)
    for i in range(array.shape[0]):
        video.write((array[i, :, :] * 255).astype(np.uint8))
    video.release()


def get_calib_info(calib_image, dim, fit_params):
    """
    Localizes PSFs in calibration image.

    Parameters
    ----------
    calib_image : np.ndarray
        Calibration image.

    dim : tuple
        Dimension of desired PSFs. Should be square (N,N).

    fit_params : dict
        Dictionary of parameters for calibration.

    Returns
    -------
    coord_list : list
        List of coordinates of PSFs in calibration image (xy) format.

    calib_image : np.ndarray
        Cropped calibration image.

    """

    psf = calib_image.copy()
    psf[psf < 0] = 0
    psf[psf < np.quantile(psf, 0.9)] = 0
    raw_coord = corner_peaks(
        erosion(psf, disk(2)),
        min_distance=fit_params["min_distance"],
        indices=True,
        threshold_rel=fit_params["threshold"],
    )
    distances = np.sqrt(np.sum(np.square(raw_coord - fit_params["sys_center"]), axis=1))
    if fit_params["centered_psf"]:
        center = raw_coord[np.argmin(distances), :]
    else:
        center = fit_params["sys_center"]
    if dim // 2 > center[0]:
        PAD = dim // 2 - center[0]
        calib_image = np.pad(calib_image, ((PAD, 0), (0, 0)))
        center[0] += PAD
    if dim // 2 > center[1]:
        PAD = dim // 2 - center[1]
        calib_image = np.pad(calib_image, ((0, 0), (PAD, 0)))
        center[1] += PAD
    if dim // 2 + center[0] > calib_image.shape[0]:
        PAD = dim // 2 + center[0] - calib_image.shape[0]
        calib_image = np.pad(calib_image, ((0, PAD), (0, 0)))
        center[0] -= PAD
    if dim // 2 + center[1] > calib_image.shape[1]:
        PAD = dim // 2 + center[1] - calib_image.shape[1]
        calib_image = np.pad(calib_image, ((0, 0), (0, PAD)))
        center[1] -= PAD

    calib_image = calib_image[
        center[0] - dim // 2 : center[0] + dim // 2,
        center[1] - dim // 2 : center[1] + dim // 2,
    ]

    coord_list = []
    for i in range(raw_coord.shape[0]):
        if (
            np.abs(raw_coord[i, 1] - center[1]) < dim // 2
            and np.abs(center[0] - raw_coord[i, 0]) < dim // 2
        ):
            coord_list += [(raw_coord[i, 1] - center[1], center[0] - raw_coord[i, 0])]

    calib_image[calib_image < 0] = 0
    calib_image[calib_image < np.quantile(calib_image, 0.99)] = 0
    calib_image = (calib_image / calib_image.sum()) * len(coord_list)

    return coord_list, calib_image


def mkdir(path):
    """
    Makes directory. Will not overwrite existing directory.

    Parameters
    ----------
    path : str
        Path to directory to be made.

    Returns
    -------
    None
    """
    if not os.path.exists(path):
        os.makedirs(path)


def rm(path, dir=False):
    """
    Removes file or directory.

    Parameters
    ----------
    path : str
        Path to file or directory to be removed.

    dir : bool, optional
        If True, removes directory. If False, removes file.

    Returns
    -------
    None
    """
    if os.path.exists(path):
        if dir:
            shutil.rmtree(path)
        else:
            os.remove(path)


def plot_mult(image_list, cmap="gray", path="", display=True):
    """
    Plots and saves a list of images stacked next to each other.

    Parameters
    ----------
    image_list : list
        List of images to be plotted.

    cmap : str, optional
        Colormap to be used for plotting.

    path : str, optional
        Path to save plot to. If empty string, plot will not be saved.

    show : bool, optional
        If True, plot will be shown. If False, plot will not be shown.

    Returns
    -------
    None

    """

    plt.figure()
    #  set plot size
    fig = plt.gcf()
    fig.set_size_inches(len(image_list) * PLT_SIZE, PLT_SIZE)

    #  code for making image and colorbar the same length
    ax = plt.gca()
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    im = ax.imshow(np.concatenate(image_list, axis=1), cmap=cmap)
    divider = make_axes_locatable(ax)
    cax = divider.new_vertical(size="5%", pad=0.05, pack_start=True)
    fig.add_axes(cax)
    fig.colorbar(im, cax=cax, orientation="horizontal")  # orientation='horizontal',

    #  display and/or saveout
    if display:
        plt.show()
    if path != "":
        plt.savefig(path, orientation="landscape", bbox_inches="tight")


def error_map(img, gt, path, display=False, cmap="magma"):
    """
    Plots the error map between two images.

    Parameters
    ----------
    img : np.ndarray
        Image to be compared to ground truth.

    gt : np.ndarray
        Ground truth image.

    path : str
        Path to save plot to.

    display : bool, optional
        If True, plot will be shown. If False, plot will not be shown.

    cmap : str, optional
        Colormap to be used for plotting.

    Returns
    -------
    error : np.ndarray
        Error map between img and gt.

    """

    error = np.abs(img - gt)
    plot_mult([error], path=path, show=display, cmap=cmap)
    return error


def show(img):
    """
    Simple plot function for grayscale images.

    Parameters
    ----------
    img : np.ndarray
        Image to be plotted.

    Returns
    -------
    None
    """
    plt.figure()
    plt.imshow(img, cmap="gray")
    plt.axis("off")
    plt.show()


def mse(im1, im2):
    """
    Computes mse between two images

    Parameters
    ----------
    im1 : np.ndarray
        First image.

    im2 : np.ndarray
        Second image.

    Returns
    -------
    mse : float
        Mean squared error between im1 and im2.

    """
    return ((im1 - im2) ** 2).mean()


def normalize(img):
    """
    Normalizes image to [0,1]

    Parameters
    ----------
    img : np.ndarray
        Image to be normalized.

    Returns
    -------
    img : np.ndarray
        Normalized image.

    """

    img = img - img.min()
    return img / img.max()


def process_video(stack, back, center, dim):
    """
    Processes a stack of images.

    Parameters
    ----------
    stack : np.ndarray
        Stack of images to be processed. Should be of shape (N, H, W).

    back : np.ndarray
        Background image to be subtracted from each image in stack.

    center : tuple
        Center of each image in stack. (i,j) format

    dim : tuple
        Desired dimension of each image in final stack.

    Returns
    -------
    new_stack : np.ndarray
        Processed stack of images.
    """

    new_stack = np.zeros((stack.shape[0], dim[0], dim[1]))
    for i in range(stack.shape[0]):
        new_stack[i, :, :] = process(stack[i, :, :], back, dim, center)
    return new_stack


def center_crop(img, des_shape, m_center=None):
    """
    Crops an image to a desired shape centered at m_center.

    Parameters
    ----------
    img : np.ndarray
        Image to be cropped.

    des_shape : tuple
        Desired shape of cropped image.

    m_center : tuple, optional
        Center of cropped image. (i,j) format. If None, center of img will be used.

    Returns
    -------
    img : np.ndarray
        Cropped image. Will be of shape des_shape.

    """

    if m_center is None:
        m_center = (img.shape[0] // 2, img.shape[1] // 2)
    left, right, up, down = (
        m_center[1] - des_shape[1] // 2,
        m_center[1] + int(np.round(des_shape[1] / 2)),
        m_center[0] - des_shape[0] // 2,
        m_center[0] + int(np.round(des_shape[0] / 2)),
    )
    # TODO: Debug this for images of an odd size.
    img = img[up:down, left:right]
    return img


def getCircList(center, radius, num_points):  # expects center = [x,y]
    """
    Produces a list of evenly-spaced points on a circle.

    Parameters
    ----------
    center : tuple
        Center of circle. (x,y) format.

    radius : int
        Radius of circle.

    num_points : int
        Number of points to be returned.

    Returns
    -------
    point_list : list
        List of points on circle. (x,y) format.

    """
    return [
        (
            int(
                np.floor(math.cos((2 * math.pi / num_points) * x) * radius + center[0])
            ),
            int(
                np.floor(math.sin((2 * math.pi / num_points) * x) * radius + center[1])
            ),
        )
        for x in range(0, num_points)
    ]


def getRadialImpulse(dim, radii, sparsity=2):
    """
    returns a radial grid of points in an array.

    Parameters
    ----------
    dim : tuple
        Dimension of desired impulse image.

    radii : list
        List of radii of circles to be drawn.

    sparsity : int, optional
        Sparsity of circles. Higher sparsity means fewer points on each circle.


    Returns
    -------
    circular_impulse : np.ndarray
        Radial impulse of given radii and sparsity.

    point_list : list
        List of points on radial impulse. (x,y) format.

    """

    point_list = []
    for r in radii:
        point_list += getCircList((dim[0] // 2, dim[0] // 2), r, r // sparsity)
    point_list = list(set(point_list))
    circular_impulse = np.zeros(dim)
    point_list += [(dim[0] // 2, dim[1] // 2)]
    for p in point_list:
        circular_impulse[p[1], p[0]] = 1
    return circular_impulse, point_list


def shift_torch(img, shift, mode="bilinear"):
    """
    Shifts an image by a given, potentially non-integer amount.

    Parameters
    ----------
    img : torch.Tensor
        Image to be shifted.

    shift : tuple
        Amount to shift image by. (i,j) format.

    mode : str, optional
        Interpolation mode. See torch.nn.functional.grid_sample for more info.

    Returns
    -------
    img : torch.Tensor
        Shifted image.
    """
    xs = np.arange(0, img.shape[1]) - shift[1]
    ys = np.arange(0, img.shape[0]) - shift[0]
    x, y = np.meshgrid(xs, ys)

    gx = 2.0 * (x / (img.shape[1] - 1)) - 1.0
    gy = 2.0 * (y / (img.shape[0] - 1)) - 1.0

    grid = torch.tensor(
        np.concatenate((gx[:, :, None], gy[:, :, None]), axis=2), device=img.device
    )

    return grid_sample(
        img[None, None, :, :].float(),
        grid[None, :, :].float(),
        padding_mode="zeros",
        mode=mode,
        align_corners=True,
    ).squeeze()


def crop_pad(img, dim):
    """
    Crops or pads a square image to a desired dimension.

    Parameters
    ----------
    img : np.ndarray
        Image to be cropped or padded.

    dim : int
        Desired sidelength of image.

    Returns
    -------
    img : np.ndarray
        Cropped or padded image. Will be of shape (dim,dim).

    """

    # crop
    if img.shape[0] > dim:
        img = center_crop(img, (dim, dim))
    # pad
    elif img.shape[0] < dim:
        pad_length = (dim - img.shape[0]) // 2
        img = np.pad(img, ((pad_length, pad_length), (pad_length, pad_length)))

    return img


def process(
    img, dim, back=None, center=None, hot_pix=False, tolerance=1, remove_outer=True
):
    """
    Processes experimental data prior to reconstruction.

    Parameters
    ----------
    img : np.ndarray
        Image to be processed.

    dim : tuple
        Desired dimension of processed image.

    back : np.ndarray, optional
        Background image to be subtracted from img.

    center : tuple, optional
        Center of img. (i,j) format. If None, center of img will be used.

    hot_pix : bool, optional
        If True, hot pixels will be removed from img.

    tolerance : float, optional
        Tolerance for hot pixel removal.

    remove_outer : bool, optional
        If True, outer ring of img will be set to 0.

    Returns
    -------
    img : np.ndarray
        Processed image. Will be of shape dim.

    """
    if center is None:
        center = (dim[0] // 2, dim[1] // 2)
    if back is not None:
        img = img - back
    img[img < 0] = 0
    img = img[
        center[0] - dim[0] // 2 : center[0] + dim[0] // 2,
        center[1] - dim[1] // 2 : center[1] + dim[1] // 2,
    ]

    if hot_pix:
        _, img = find_outlier_pixels(img, tolerance=tolerance)

    if remove_outer:
        img = outer_ring(img)

    return normalize(img)


def outer_ring(image):
    """
    Sets outer ring of image to 0.

    Parameters
    ----------
    image : np.ndarray
        Image to be processed.

    Returns
    -------
    image : np.ndarray
        Processed image.
    """
    image[0, :] = 0
    image[-1, :] = 0
    image[:, 0] = 0
    image[:, -1] = 0

    return image


# credit: https://stackoverflow.com/questions/18951500/automatically-remove-hot-dead-pixels-from-an-image-in-python
def find_outlier_pixels(data, tolerance=3, worry_about_edges=True):
    """
    Finds hot pixels in an image.

    Parameters
    ----------
    data : np.ndarray
        Image to be processed.

    tolerance : float, optional
        Tolerance for hot pixel removal.

    worry_about_edges : bool, optional
        If True, hot pixels on edges of image will be removed.

    Returns
    -------
    hot_pixels : list
        List of coordinates of hot pixels. (i,j) format.

    fixed_image : np.ndarray
        Image with hot pixels removed.

    Notes
    -----
    This function was taken from https://stackoverflow.com/questions/18951500/automatically-remove-hot-dead-pixels-from-an-image-in-python
    If you want to ignore the edges and greatly speed up the code, then set
    worry_about_edges to False.
    """

    blurred = median_filter(data, size=4)
    difference = data - blurred
    threshold = tolerance * np.std(difference)

    # find the hot pixels, but ignore the edges
    hot_pixels = np.nonzero((np.abs(difference[1:-1, 1:-1]) > threshold))
    hot_pixels = (
        np.array(hot_pixels) + 1
    )  # because we ignored the first row and first column

    fixed_image = np.copy(data)  # This is the image with the hot pixels removed
    for y, x in zip(hot_pixels[0], hot_pixels[1]):
        fixed_image[y, x] = blurred[y, x]

    if worry_about_edges is True:
        height, width = np.shape(data)

        ###Now get the pixels on the edges (but not the corners)###

        # left and right sides
        for index in range(1, height - 1):
            # left side:
            med = np.median(data[index - 1 : index + 2, 0:2])
            diff = np.abs(data[index, 0] - med)
            if diff > threshold:
                hot_pixels = np.hstack((hot_pixels, [[index], [0]]))
                fixed_image[index, 0] = med

            # right side:
            med = np.median(data[index - 1 : index + 2, -2:])
            diff = np.abs(data[index, -1] - med)
            if diff > threshold:
                hot_pixels = np.hstack((hot_pixels, [[index], [width - 1]]))
                fixed_image[index, -1] = med

        # Then the top and bottom
        for index in range(1, width - 1):
            # bottom:
            med = np.median(data[0:2, index - 1 : index + 2])
            diff = np.abs(data[0, index] - med)
            if diff > threshold:
                hot_pixels = np.hstack((hot_pixels, [[0], [index]]))
                fixed_image[0, index] = med

            # top:
            med = np.median(data[-2:, index - 1 : index + 2])
            diff = np.abs(data[-1, index] - med)
            if diff > threshold:
                hot_pixels = np.hstack((hot_pixels, [[height - 1], [index]]))
                fixed_image[-1, index] = med

        ###Then the corners###

        # bottom left
        med = np.median(data[0:2, 0:2])
        diff = np.abs(data[0, 0] - med)
        if diff > threshold:
            hot_pixels = np.hstack((hot_pixels, [[0], [0]]))
            fixed_image[0, 0] = med

        # bottom right
        med = np.median(data[0:2, -2:])
        diff = np.abs(data[0, -1] - med)
        if diff > threshold:
            hot_pixels = np.hstack((hot_pixels, [[0], [width - 1]]))
            fixed_image[0, -1] = med

        # top left
        med = np.median(data[-2:, 0:2])
        diff = np.abs(data[-1, 0] - med)
        if diff > threshold:
            hot_pixels = np.hstack((hot_pixels, [[height - 1], [0]]))
            fixed_image[-1, 0] = med

        # top right
        med = np.median(data[-2:, -2:])
        diff = np.abs(data[-1, -1] - med)
        if diff > threshold:
            hot_pixels = np.hstack((hot_pixels, [[height - 1], [width - 1]]))
            fixed_image[-1, -1] = med

    return hot_pixels, fixed_image


def get_psf_extent(psf):
    """
    Gets the approximate diameter of a PSF.

    Parameters
    ----------
    psf : np.ndarray
        PSF to be measured. Assumed to be square

    Returns
    -------
    extent : int
        Approximate diameter of PSF in pixels

    """

    center = psf.shape[0] // 2  # assumes psf is square shaped
    psf_slice = psf[center, center:]
    psf_slice[psf_slice < torch.quantile(psf_slice, 0.9)] = 0
    return max(psf_slice.nonzero()) * 2


def tensor_to_np(tensor):
    """
    Converts a torch.Tensor on gpu to a np.ndarray on cpu

    Parameters
    ----------
    tensor : torch.Tensor
        Tensor to be converted

    Returns
    -------
    img : np.ndarray
        Converted image
    """
    img = tensor.cpu().detach().numpy().squeeze(0)
    return img


def load_model(model_type, model_path, device):
    """
    Loads a deep learing model from a given path

    Parameters
    ----------
    model_type : torch.nn.Module
        Type of model to be loaded.

    model_path : str
        Path to model weights to be loaded.

    device : torch.device
        Device to load model to.

    Returns
    -------
    model_type : torch.nn.Module
        Loaded model.

    """
    # from previous section num_epochs
    model_type.load_state_dict(torch.load(model_path, map_location=device))
    model_type.eval()
    return model_type  # returns a loaded model
