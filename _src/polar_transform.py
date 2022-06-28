
import numpy as np
import pdb
import torch
import pathlib
import torch.nn.functional as fun
#import cv2
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
#from src.ddn.ddn.pytorch.node import *
from ._src import util
import pdb

dirname = str(pathlib.Path(__file__).parent.absolute())


def getCartesianPoints(r, theta, center):
    """ Convert list of polar points to cartesian points """
    x = r * np.cos(theta) + center[0]
    y = r * np.sin(theta) + center[1]

    return x, y


def getPolarPoints(x, y, center):
    """Convert list of cartesian points to polar points"""
    cX, cY = x - center[0], y - center[1]

    r = np.sqrt(cX ** 2 + cY ** 2)

    theta = np.arctan2(cY, cX)

    # Make range of theta 0 -> 2pi instead of -pi -> pi
    # According to StackOverflow, this is the fastest method:
    # https://stackoverflow.com/questions/37358016/numpy-converting-range-of-angles-from-pi-pi-to-0-2pi
    theta = np.where(theta < 0, theta + 2 * np.pi, theta)

    return r, theta

def img2polar(img, initialRadius=0, numRadii=None , finalRadius=None, initialAngle=0, finalAngle=np.pi*2, center=None, border='constant'):
    if center == None:
        center = (np.array(img.shape[-1:-3:-1])-1) / 2.0

    if numRadii is None:
        numRadii = max(img.shape)

    if finalRadius is None:
        corners = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]) * img.shape[-2:]
        radii, _ = getPolarPoints(corners[:, 1], corners[:, 0], center)
        maxRadius = np.ceil(radii.max()).astype(int)
        finalRadius = (np.sqrt(2) * (np.array(img.shape[-1:-3:-1]) // 2))[0]


    if numRadii > 500:
        numAngles = int(2 * np.max(img.shape) * ((finalAngle - initialAngle) / (2 * np.pi)))
    else:
        numAngles = int(4 * np.max(img.shape) * ((finalAngle - initialAngle) / (2 * np.pi)))

    radii = np.linspace(initialRadius, finalRadius, numRadii, endpoint=False)
    theta = np.linspace(initialAngle, finalAngle, numAngles, endpoint=False)
    r, theta = np.meshgrid(radii, theta)

    xCartesian, yCartesian = getCartesianPoints(r, theta, center)

    # Reshape the image to be 3D, flattens the array if > 3D otherwise it makes it 3D with the 3rd dimension a size of 1
    image = img.reshape((-1,) + img.shape)

    
   # image = fun.pad(image[None, :], (3, 3, 3, 3), 'replicate')[0,:,:,:]

    if border == 'constant':
        # Pad image by 3 pixels and then offset all of the desired coordinates by 3
        image = fun.pad(image[None, :], (3, 3, 3, 3), 'replicate')[0,:,:,:]
        xCartesian += 3
        yCartesian += 3

    gx = 2.0 * (xCartesian / (image.shape[2] - 1)) - 1.
    gy = 2.0 * (yCartesian / (image.shape[1] - 1)) - 1.
    desiredCoords = torch.tensor(np.stack((gx, gy), 2), device=img.device).unsqueeze(0)

    polarImage = fun.grid_sample(image[None,:], desiredCoords.float(), padding_mode='zeros', mode='bicubic', align_corners=True)
    polarImage = polarImage.squeeze()

    return polarImage

def polar2img(img, imageSize, initialRadius=0, finalRadius=None, initialAngle=0, finalAngle=np.pi*2, center=None, border='constant'):
    if center == None:
        #center = (np.array(imageSize) / 2).astype(int)
        #center = (191.5, 191.5)
        #center = (192, 192)
        center = ((imageSize[0]-1)/2.0, (imageSize[1])/2.0)
        #center = (np.array(imageSize[-1:-3:-1])-1) / 2.0

    if finalRadius is None:
        corners = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]) * imageSize[-2:]
        radii, _ = getPolarPoints(corners[:, 1], corners[:, 0], center)
        #finalRadius = np.ceil(radii.max()).astype(int)
        finalRadius = (np.sqrt(2) * imageSize[0]//2)
    # This is used to scale the result of the radius to get the appropriate Cartesian value
    scaleRadius = img.shape[1] / (finalRadius - initialRadius)

    # This is used to scale the result of the angle to get the appropriate Cartesian value
    scaleAngle = img.shape[0] / (finalAngle - initialAngle)

    # Get list of cartesian x and y coordinate and create a 2D create of the coordinates using meshgrid
    xs = np.arange(0, imageSize[1])
    ys = np.arange(0, imageSize[0])
    x, y = np.meshgrid(xs, ys)

    # Take cartesian grid and convert to polar coordinates
    r, theta = getPolarPoints(x, y, center)

    # Offset the radius by the initial source radius
    r = r - initialRadius

    # Offset the theta angle by the initial source angle
    # The theta values may go past 2pi, so they are looped back around by taking modulo with 2pi.
    # Note: This assumes initial source angle is positive
    theta = np.mod(theta - initialAngle + 2 * np.pi, 2 * np.pi)

    # Scale the radius using scale factor
    r = r * scaleRadius

    # Scale the angle from radians to pixels using scale factor
    theta = theta * scaleAngle

    # Reshape the image to be 3D, flattens the array if > 3D otherwise it makes it 3D with the 3rd dimension a size of 1
    image = img.reshape((-1,) + img.shape)

    if border == 'constant':
        # Pad image by 3 pixels and then offset all of the desired coordinates by 3
        image = fun.pad(image[None, :], (3, 3, 3, 3), 'replicate')[0,:,:,:]
        r += 3
        theta += 3

    gr = 2.0 * (r / (image.shape[2] - 1)) - 1.
    gtheta = 2.0 * (theta / (image.shape[1] - 1)) - 1.
    desiredCoords = torch.tensor(np.stack((gr, gtheta), 2), device=img.device).unsqueeze(0)

    cartImage = fun.grid_sample(image[None,:], desiredCoords.float(), padding_mode='zeros', mode='bicubic', align_corners=True)
    cartImage = cartImage.squeeze()
    return cartImage

if __name__ == "__main__":

    # np_im = plt.imread('../data/test_images/baboon.png')
    # grid_x, grid_y = np.meshgrid(np.arange(np_im.shape[1]), np.arange(np_im.shape[0]))
    # dx = 2 * np.random.randn(*np_im.shape[:-1])
    # dy = 2 * np.random.randn(*np_im.shape[:-1])
    # def_grid_x = grid_x + dx
    # def_grid_y = grid_y + dy
    #
    # print('scipy')
    # out_im_r = scipy.ndimage.map_coordinates(np_im[:, :, 0], [def_grid_y, def_grid_x], order=1); \
    # out_im_g = scipy.ndimage.map_coordinates(np_im[:, :, 1], [def_grid_y, def_grid_x], order=1); \
    # out_im_b = scipy.ndimage.map_coordinates(np_im[:, :, 2], [def_grid_y, def_grid_x], order=1); \
    # out_im = np.stack((out_im_r, out_im_g, out_im_b), 2)
    #
    #
    # torch_im = torch.from_numpy(np.float64(np_im)).unsqueeze(0).permute(0, 3, 1, 2)
    # gx = 2.0 * def_grid_x / (np_im.shape[1] - 1) - 1.
    # gy = 2.0 * def_grid_y / (np_im.shape[0] - 1) - 1.
    #
    #
    # torch_grid = torch.from_numpy(np.stack((gx, gy), 2)).unsqueeze(0)
    # torch_out = fun.grid_sample(torch_im, torch_grid, mode='bilinear', align_corners=False,
    #                             padding_mode='reflection').squeeze().permute(1, 2, 0)

    #Read in image and convert to tensor for ground truth
    im = rgb2gray(plt.imread('test_images/baboon.png'))
    #Im = cv2.resize(Im, dsize=(384, 384))
    im = torch.Tensor(im)
    #Im /= Im.sum()

    polar_im = img2polar(im)
    recon = polar2img(polar_im, im.shape)

    histogram, bin_edges = np.histogram(im, bins=256, range=(0.1, 1))
    histogram_polar, bin_edges_polar = np.histogram(polar_im, bins=256, range=(0.1, 1))
    histogram_recon, bin_edges_recon = np.histogram(recon, bins=256, range=(0.1, 1))



    plt.plot(bin_edges[0:-1], histogram, color= 'r')
    plt.plot(bin_edges_polar[0:-1], histogram_polar)
    plt.plot(bin_edges_recon[0:-1], histogram_recon)
    plt.show()

    error = torch.norm(im-recon)

    print(error)

    pdb.set_trace()

    # #make copy to feed in for the forward pass
    # I = Im.detach().clone()
    # print(f"sum_before: {I.sum()}")
    # I.requires_grad = True
    # #po, settings = pt.convertToPolarImage(I.detach().numpy(), order=1, border='constant')  # theirs
    # pot = img2polar(I)

    # util.plot(pot.detach())

    # I_recon = polar2img(pot, I.shape)
    # print(f"sum_after: {I_recon.sum()}")
    # util.plot(I_recon.detach())
    # loss = torch.norm(Im - I_recon)
    # print(f"loss: {loss}")
    # loss.backward()
    # print(f"gradient: {I.grad}")
