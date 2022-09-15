
import numpy as np
import pdb
import torch
import pathlib
import torch.nn.functional as fun
import cv2
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from . import util
import scipy

dirname = str(pathlib.Path(__file__).parent.absolute())

INTERP_TYPE = 'bilinear'
 

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

def img2polar(img, numRadii=None, initialRadius=0, finalRadius=None, initialAngle=0, finalAngle=np.pi*2, center=None, border='constant', ispsf=False):
    # if ispsf:
    #     INTERP_TYPE = 'bicubic'
    # else:
    #     INTERP_TYPE = 'bicubic'
    
    if center == None:
        center = (np.array(img.shape[-1:-3:-1])-1) / 2.0


    if finalRadius is None:
        corners = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]) * img.shape[-2:]
        radii, _ = getPolarPoints(corners[:, 1], corners[:, 0], center)
        maxRadius = np.ceil(radii.max()).astype(int)
        #=finalRadius = (np.sqrt(2) * ((img.shape[0]-1)/2))
        finalRadius = radii.max()

    
    initialRadius = getPolarPoints(img.shape[0]//2,img.shape[0]//2, center)[0]

    maxSize = np.max(img.shape)
    if numRadii is None:
        numRadii = maxSize
    
    initialAngle = np.pi/4
    finalAngle = 2*np.pi + np.pi/4

    if maxSize > 500:
        numAngles = int(2 * np.max(img.shape) * ((finalAngle - initialAngle) / (2 * np.pi)))
    else:
        numAngles = int(4 * np.max(img.shape) * ((finalAngle - initialAngle) / (2 * np.pi)))

    numAngles = int(5 * np.max(img.shape)) #- 4

    radii = np.linspace(initialRadius, finalRadius, numRadii, endpoint=False)
    #radii = (np.sqrt(2)*(np.linspace(0, img.shape[0]/2, numRadii, endpoint=False) + 0.5))
    
    theta = np.linspace(initialAngle, finalAngle, numAngles, endpoint=False)
    r, theta = np.meshgrid(radii, theta)

    xCartesian, yCartesian = getCartesianPoints(r, theta, center)
    # Reshape the image to be 3D, flattens the array if > 3D otherwise it makes it 3D with the 3rd dimension a size of 1
    image = img.reshape((-1,) + img.shape)

    pad = 3
    if border == 'constant':
        # Pad image by 3 pixels and then offset all of the desired coordinates by 3
        image = fun.pad(image[None, :], (pad, pad, pad, pad), 'replicate')[0,:,:,:]
        xCartesian += pad
        yCartesian += pad

    gx = 2.0 * (xCartesian / (image.shape[2] - 1)) - 1.
    gy = 2.0 * (yCartesian / (image.shape[1] - 1)) - 1.
    desiredCoords = torch.from_numpy(np.stack((gx, gy), 2)).unsqueeze(0).to(device=img.device)

    polarImage = fun.grid_sample(image[None,:], desiredCoords.float(), padding_mode='zeros', mode=INTERP_TYPE, align_corners=True)
    polarImage = polarImage.squeeze()

    return polarImage

def polar2img(img, imageSize=None, initialRadius=0, finalRadius=None, initialAngle=0, finalAngle=np.pi*2, center=None, border='constant', ispsf=False):
    # if ispsf:
    #     INTERP_TYPE = 'bicubic'
    # else:
    #     INTERP_TYPE = 'bicubic'
    if center == None:
        center = ((imageSize[0]-1)/2.0, (imageSize[1]-1)/2.0)

    initialAngle = np.pi/4
    finalAngle = 2*np.pi + np.pi/4
    initialRadius = getPolarPoints(imageSize[0]//2,imageSize[1]//2, center)[0]

    if finalRadius is None:
        corners = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]) * imageSize[-2:]
        radii, _ = getPolarPoints(corners[:, 1], corners[:, 0], center)
        finalRadius = np.ceil(radii.max()).astype(int)
        finalRadius = radii.max()
        #initialRadius = np.sqrt(2)/2
        #finalRadius = (np.sqrt(2) * ((imageSize[0]-1)/2))
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

    pad = 3

    if border == 'constant':
        # Pad image by 3 pixels and then offset all of the desired coordinates by 3
        image = fun.pad(image[None, :], (pad, pad, pad, pad), 'replicate')[0,:,:,:]
        r += pad
        theta += pad


    gr = 2.0 * (r / (image.shape[2] - 1)) - 1.
    gtheta = 2.0 * (theta / (image.shape[1] - 1)) - 1.
    desiredCoords = torch.from_numpy(np.stack((gr, gtheta), 2)).unsqueeze(0).to(device=img.device)

    cartImage = fun.grid_sample(image[None,:], desiredCoords.float(), padding_mode='zeros', mode=INTERP_TYPE, align_corners=True)
    cartImage = cartImage.squeeze()
    cartImage[cartImage<0]= 0
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
    Im = rgb2gray(plt.imread('../data/test_images/baboon.png'))
    Im = cv2.resize(Im, dsize=(384, 384))
    Im = torch.Tensor(Im)
    Im /= Im.sum()

    #make copy to feed in for the forward pass
    I = Im.detach().clone()
    print(f"sum_before: {I.sum()}")
    I.requires_grad = True
    #po, settings = pt.convertToPolarImage(I.detach().numpy(), order=1, border='constant')  # theirs
    pot = img2polar(I)

    util.plot(pot.detach())

    I_recon = polar2img(pot, I.shape)
    print(f"sum_after: {I_recon.sum()}")
    util.plot(I_recon.detach())
    loss = torch.norm(Im - I_recon)
    print(f"loss: {loss}")
    loss.backward()
    print(f"gradient: {I.grad}")