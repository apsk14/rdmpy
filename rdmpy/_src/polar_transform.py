"""
Impelmetation of polar transform for pytorch. This is a modified version of the 
polar transform from the following repository: https://github.com/addisonElliott/polarTransform
"""

import pathlib

import numpy as np
import torch
import torch.nn.functional as fun

dirname = str(pathlib.Path(__file__).parent.absolute())

INTERP_TYPE = "bicubic"


def getCartesianPoints(r, theta, center):
    """
    Convert list of polar points to cartesian points

    Parameters
    ----------
    r : np.ndarray
        List of radii

    theta : np.ndarray
        List of angles

    center : tuple
        Center of the image, (x,y) format

    Returns
    -------
    x : np.ndarray
        List of x coordinates

    y : np.ndarray
        List of y coordinates

    """
    x = r * np.cos(theta) + center[0]
    y = r * np.sin(theta) + center[1]

    return x, y


def getPolarPoints(x, y, center):
    """
    Convert list of cartesian points to polar points

    Parameters
    ----------
    x : np.ndarray
        List of x coordinates

    y : np.ndarray
        List of y coordinates

    center : tuple
        Center of the image (x,y) format

    Returns
    -------
    r : np.ndarray
        List of radii

    theta : np.ndarray
        List of angles

    """
    cX, cY = x - center[0], y - center[1]

    r = np.sqrt(cX**2 + cY**2)

    theta = np.arctan2(cY, cX)

    # Make range of theta 0 -> 2pi instead of -pi -> pi
    # According to StackOverflow, this is the fastest method:
    # https://stackoverflow.com/questions/37358016/numpy-converting-range-of-angles-from-pi-pi-to-0-2pi
    theta = np.where(theta < 0, theta + 2 * np.pi, theta)

    return r, theta


def img2polar(
    img,
    numRadii=None,
    finalRadius=None,
    initialAngle=0,
    finalAngle=np.pi * 2,
    center=None,
    border="constant",
):
    """
    Converts cartesian image to polar image

    Parameters
    ----------
    img : torch.Tensor
        Image to be converted. Should be (B,N,N)

    numRadii : int, optional
        Number of radii to use. If None, the image sidelength is used.

    finalRadius : float, optional
        Final radius to use. If None, the image diagonal is used.

    initialAngle : float, optional
        Initial angle to use in radians. Default is 0.

    finalAngle : float, optional
        Final angle to use in radians. Default is 2pi.

    center : tuple, optional
        Center of the image (x,y) format. Default is the center of the image.

    border : str, optional
        How to handle borders. Options are 'constant', 'reflect', 'replicate',
        'circular', 'zeros'. Default is 'constant'.

    Returns
    -------
    polarImage : torch.Tensor
        Polar image will be (numAngles, numRadii)

    """

    if center is None:
        center = (np.array(img.shape[-1:-3:-1]) - 1) / 2.0

    if finalRadius is None:
        corners = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]) * img.shape[-2:]
        radii, _ = getPolarPoints(corners[:, 1], corners[:, 0], center)
        finalRadius = radii.max()
        finalRadius = (img.shape[0] / 2) * np.sqrt(2)

    maxSize = np.max(img.shape)
    if numRadii is None:
        numRadii = maxSize

    initialAngle = np.pi / 4
    finalAngle = 2 * np.pi + np.pi / 4

    if maxSize > 700:
        numAngles = int(
            2 * np.max(img.shape) * ((finalAngle - initialAngle) / (2 * np.pi))
        )
    else:
        numAngles = int(
            4 * np.max(img.shape) * ((finalAngle - initialAngle) / (2 * np.pi))
        )

    radii = np.sqrt(2) * (
        np.linspace(0, (img.shape[-2] / 2), numRadii, endpoint=False, retstep=False)
        + 0.5
    )

    theta = np.linspace(initialAngle, finalAngle, numAngles, endpoint=False)
    r, theta = np.meshgrid(radii, theta)

    xCartesian, yCartesian = getCartesianPoints(r, theta, center)
    # Reshape the image to be 3D, flattens the array if > 3D otherwise it makes it 3D
    if len(img.shape) < 3:
        image = img.reshape((-1,) + img.shape)
    else:
        image = img

    pad = 3
    if border == "constant":
        # Pad image by 3 pixels and then offset all of the desired coordinates by 3
        image = fun.pad(image[None, :], (pad, pad, pad, pad), "replicate")[0, :, :, :]
        xCartesian += pad
        yCartesian += pad

    gx = 2.0 * (xCartesian / (image.shape[2] - 1)) - 1.0
    gy = 2.0 * (yCartesian / (image.shape[1] - 1)) - 1.0
    desiredCoords = (
        torch.from_numpy(np.stack((gx, gy), 2)).unsqueeze(0).to(device=img.device)
    )

    polarImage = fun.grid_sample(
        image[None, :],
        desiredCoords.float(),
        padding_mode="zeros",
        mode=INTERP_TYPE,
        align_corners=True,
    )
    polarImage = polarImage.squeeze()

    return polarImage


def polar2img(
    img,
    imageSize=None,
    initialRadius=0,
    finalRadius=None,
    initialAngle=0,
    finalAngle=np.pi * 2,
    center=None,
    border="constant",
):
    """
    Converts polar image to cartesian image

    Parameters
    ----------
    img : torch.Tensor
        Polar mage to be converted. Should be (A,R) where A is angles and R is radii.

    imageSize : tuple, optional
        Size of the image to be returned. If None, the image is assumed to be (R,R).

    initialRadius : float, optional
        Initial radius to use. Default is 0.

    finalRadius : float, optional
        Final radius to use. If None, the image diagonal is used.

    initialAngle : float, optional
        Initial angle to use in radians. Default is 0.

    finalAngle : float, optional
        Final angle to use in radians. Default is 2pi.

    center : tuple, optional
        Center of the image (x,y) format. Default is the center of the image.

    border : str, optional
        How to handle borders. Options are 'constant', 'reflect', 'replicate',

    Returns
    -------
    cartImage : torch.Tensor
        Cartesian image, will be imageSize

    """

    if center is None:
        center = ((imageSize[-2] - 1) / 2.0, (imageSize[-1] - 1) / 2.0)

    initialAngle = np.pi / 4
    finalAngle = 2 * np.pi + np.pi / 4
    # initialRadius = getPolarPoints(imageSize[0]//2,imageSize[1]//2, center)[0]

    if finalRadius is None:
        corners = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]) * imageSize[-2:]
        radii, _ = getPolarPoints(corners[:, 1], corners[:, 0], center)

    # This is used to scale the result of the radius to get the Cartesian value
    radii = np.sqrt(2) * (
        np.linspace(
            0, (imageSize[-2] / 2), img.shape[-1], endpoint=False, retstep=False
        )
        + 0.5
    )
    initialRadius = radii[0]
    finalRadius = radii[-1]
    scaleRadius = img.shape[-1] / (finalRadius - initialRadius)

    # This is used to scale the result of the angle to get the  Cartesian value
    scaleAngle = img.shape[-2] / (finalAngle - initialAngle)

    # Get list of cartesian x and y coordinate and create a 2D create of the coordinates
    xs = np.arange(0, imageSize[-1])
    ys = np.arange(0, imageSize[-2])
    x, y = np.meshgrid(xs, ys)

    # Take cartesian grid and convert to polar coordinates
    r, theta = getPolarPoints(x, y, center)

    # Offset the radius by the initial source radius
    r = r - initialRadius

    # Offset the theta angle by the initial source angle
    # The theta values may go past 2pi, so they are modulo 2pi.
    # Note: This assumes initial source angle is positive
    theta = np.mod(theta - initialAngle + 2 * np.pi, 2 * np.pi)

    # Scale the radius using scale factor
    r = r * scaleRadius

    # Scale the angle from radians to pixels using scale factor
    theta = theta * scaleAngle

    # Reshape the image to be 3D, flattens the array if > 3D otherwise it makes it 3D
    if len(img.shape) < 3:
        image = img.reshape((-1,) + img.shape)
    else:
        image = img

    pad = 3

    if border == "constant":
        # Pad image by 3 pixels and then offset all of the desired coordinates by 3
        image = fun.pad(image[None, :], (pad, pad, pad, pad), "replicate")[0, :, :, :]
        r += pad
        theta += pad

    gr = 2.0 * (r / (image.shape[2] - 1)) - 1.0
    gtheta = 2.0 * (theta / (image.shape[1] - 1)) - 1.0
    desiredCoords = (
        torch.from_numpy(np.stack((gr, gtheta), 2)).unsqueeze(0).to(device=img.device)
    )

    cartImage = fun.grid_sample(
        image[None, :],
        desiredCoords.float(),
        padding_mode="zeros",
        mode=INTERP_TYPE,
        align_corners=True,
    )
    cartImage = cartImage.squeeze()
    return cartImage
