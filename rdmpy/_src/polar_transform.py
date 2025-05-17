"""
Impelmetation of polar transform for pytorch. This is a modified version of the
polar transform from the following repository: https://github.com/addisonElliott/polarTransform
"""

import pathlib

import numpy as np
import torch
import torch.nn.functional as fun

import pdb

dirname = str(pathlib.Path(__file__).parent.absolute())

INTERP_TYPE = "bicubic"


def polar_shift_alt(img, shifts, center=None):
    if center is None:
        center = [(img.shape[1] - 1) / 2.0, (img.shape[1] - 1) / 2.0]

    initialAngle = np.pi / 4
    finalAngle = 2 * np.pi + np.pi / 4

    numRadii = img.shape[1]
    numAngles = img.shape[0]

    # finalRadius = np.sqrt(2) * (numRadii / 2)
    # initialRadius = 0.0

    # Step 1: Generate polar sampling grid
    # theta = torch.linspace(initialAngle, finalAngle, steps=numAngles, device=img.device)
    # radii = torch.linspace(
    #     initialRadius, finalRadius, steps=numRadii, device=img.device
    # )

    radii = np.sqrt(2) * (
        torch.linspace(
            torch.tensor(0.0),
            ((img.shape[-2] - 1) / torch.tensor(2.0)),
            numRadii,
            device=img.device,
        )
        + 0.5
    )

    theta = torch.linspace(initialAngle, finalAngle, numAngles, device=img.device)
    r, theta = torch.meshgrid(radii, theta, indexing="xy")

    xCartesian, yCartesian = getCartesianPointsTorch(r, theta, center)

    xCartesian = xCartesian - shifts[0] * 4.0
    yCartesian = yCartesian - shifts[0] * 4.0

    scaleRadius = img.shape[1] / (radii[-1] - radii[0])
    scaleAngle = img.shape[0] / (finalAngle - initialAngle)

    r_new, theta_new = getPolarPointsTorch(xCartesian, yCartesian, center)

    # Offset the radius by the initial source radius
    r = r_new - radii[0]

    # Offset the theta angle by the initial source angle
    # The theta values may go past 2pi, so they are modulo 2pi.
    # Note: This assumes initial source angle is positive
    theta = torch.remainder(theta_new - initialAngle + 2 * np.pi, 2 * np.pi)

    # Scale the radius using scale factor
    r = r * scaleRadius

    # Scale the angle from radians to pixels using scale factor
    theta = theta * scaleAngle

    pad = 10

    # Pad image by 3 pixels and then offset all of the desired coordinates by 3
    img = fun.pad(img[None, :, :], (pad, pad, pad, pad), "replicate").squeeze()
    r += pad
    theta += pad

    gr = 2.0 * (r / (img.shape[-1] - 1)) - 1.0
    gtheta = 2.0 * (theta / (img.shape[-2] - 1)) - 1.0
    desiredCoords = torch.stack((gr, gtheta), 2)

    cartImage = fun.grid_sample(
        img[None, None, :, :].float(),
        desiredCoords[None, :, :, :].float(),
        padding_mode="zeros",
        mode="bilinear",
        align_corners=True,
    )

    return cartImage.squeeze()


def polar_shift(
    polar_img,
    shifts,
    center=None,
    finalRadius=None,
    initialAngle=np.pi / 4,
    finalAngle=2 * np.pi + np.pi / 4,
):
    """
    Shift a polar image by a Cartesian (x, y) offset, internally computing the polar grid.

    Parameters:
        polar_img : (A, R) torch.Tensor — Input polar image
        shift_x, shift_y : float — Cartesian shift to simulate (pixels)
        center : (x, y) tuple — Cartesian center of the original image
        finalRadius : float — Maximum radius used in polar transform
        initialAngle, finalAngle : float — Angle range for polar image (radians)

    Returns:
        shifted_polar : (A, R) torch.Tensor — Shifted polar image
    """

    device = polar_img.device
    A, R = polar_img.shape
    N = shifts.shape[0]

    center = [(R - 1) / 2.0, (R - 1) / 2.0] if center is None else center
    finalRadius = np.sqrt(2) * (R / 2)
    initialRadius = 0.0

    # Step 1: Generate polar sampling grid
    theta = torch.linspace(initialAngle, finalAngle, steps=A, device=device)
    radius = torch.linspace(initialRadius, finalRadius, steps=R, device=device)
    radius_old, theta_grid = torch.meshgrid(radius, theta, indexing="xy")

    # # Step 2: Convert polar → Cartesian
    x = radius_old * torch.cos(theta_grid)  # + center[0]
    y = radius_old * torch.sin(theta_grid)  # + center[1]

    # # Step 3: Apply inverse Cartesian shift
    x = x.unsqueeze(0) - shifts[:, None, None]
    y = y.unsqueeze(0) - shifts[:, None, None]

    # # Step 4: Convert back to polar
    # # x = x - center[0]
    # # y = y - center[1]
    radius_grid = torch.sqrt(x**2 + y**2)
    theta_grid = torch.atan2(y, x)

    theta_grid = (
        (theta_grid - initialAngle) % (2 * np.pi)
    ) + initialAngle  # Ensure theta is in [0, 2π)

    # Step 5: Normalize coordinates for grid_sample
    theta_grid = (theta_grid - initialAngle) / (finalAngle - initialAngle) * 2 - 1
    radius_grid = (radius_grid - initialRadius) / (finalRadius - initialRadius) * 2 - 1

    # pad = 3
    # polar_img = fun.pad(polar_img[None, :, :], (pad, pad, pad, pad), "replicate")
    # theta_grid += pad
    # radius_grid += pad

    # radius_grid = 2.0 * (radius_grid / (polar_img.shape[2] - 1)) - 1.0
    # theta_grid = 2.0 * (theta_grid / (polar_img.shape[1] - 1)) - 1.0

    grid = torch.stack((radius_grid, theta_grid), dim=-1)  # shape (B, A, R, 2)

    # Step 6: grid_sample from polar image
    polar_img_batched = polar_img.unsqueeze(0)  # .unsqueeze(0)  # (1, 1, A, R)
    shifted = fun.grid_sample(
        polar_img_batched.repeat(N, 1, 1, 1),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )

    # eps = 1e-8
    # radius_grid = radius_grid.clamp(min=eps)
    # jacobian_correction = (
    #     (radius_old / radius_grid).unsqueeze(0).unsqueeze(0)
    # )  # shape [1, 1, H, W]
    # shifted = shifted * jacobian_correction
    return shifted.squeeze()


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
    a_sampling=4,
):
    """
    Convert cartesian image to polar image

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

    a_sampling : int, optional
        How many more times to sample angle than radius. Default is 4.

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
            a_sampling * np.max(img.shape) * ((finalAngle - initialAngle) / (2 * np.pi))
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
        Polar image to be converted. Should be (A,R) where A is angles and R is radii.

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


######## Batched & Accelerated #######


def getCartesianPointsTorch(r, theta, center):
    """
    Convert list of polar points to cartesian points

    Parameters
    ----------
    r : torch.tensor
        List of radii

    theta : torch.tensor
        List of angles

    center : tuple
        Center of the image, (x,y) format

    Returns
    -------
    x : torch.tensor
        List of x coordinates

    y : torch.tensor
        List of y coordinates

    """
    x = r * torch.cos(theta) + center[0]
    y = r * torch.sin(theta) + center[1]
    return x, y


def getPolarPointsTorch(x, y, center):
    """
    Convert list of cartesian points to polar points in pytorch

    Parameters
    ----------
    x : torch.tensor
        List of x coordinates

    y : torch.tensor
        List of y coordinates

    center : tuple
        Center of the image (x,y) format

    Returns
    -------
    r : torch.tensor
        List of radii

    theta : torch.tensor
        List of angles

    """
    cX, cY = x - center[0], y - center[1]

    r = torch.sqrt(cX**2 + cY**2)

    theta = torch.atan2(cY, cX)

    # Make range of theta 0 -> 2pi instead of -pi -> pi
    # According to StackOverflow, this is the fastest method:
    # https://stackoverflow.com/questions/37358016/numpy-converting-range-of-angles-from-pi-pi-to-0-2pi
    theta = torch.where(theta < 0, theta + 2 * np.pi, theta)

    return r, theta


def batchimg2polar(
    img,
    numRadii=None,
    finalRadius=None,
    initialAngle=0,
    finalAngle=np.pi * 2,
    center=None,
    border="constant",
    a_sampling=4,
):
    """
    Converts batch of cartesian images to polar image.
    Expects multichan images, but chan & batch dimension can be 1.

    Parameters
    ----------
    img : torch.Tensor
        Image to be converted. Should be (B, C, N, N)

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
        Polar image will be (B, numAngles, numRadii, C)

    """
    if center is None:
        center = (np.array(img.shape[-1:-3:-1]) - 1) / 2.0

    if finalRadius is None:
        corners = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]) * img.shape[-2:]
        radii, _ = getPolarPoints(corners[:, 1], corners[:, 0], center)
        finalRadius = radii.max()
        finalRadius = (img.shape[2] / 2) * np.sqrt(2)

    maxSize = np.max(img.shape[2:])
    if numRadii is None:
        numRadii = maxSize

    initialAngle = np.pi / 4
    finalAngle = 2 * np.pi + np.pi / 4

    if maxSize > 700:
        numAngles = int(
            a_sampling
            * np.max(img.shape[2:])
            * ((finalAngle - initialAngle) / (2 * np.pi))
        )
    else:
        numAngles = int(
            4 * np.max(img.shape[2:]) * ((finalAngle - initialAngle) / (2 * np.pi))
        )

    # get radii and angles for generating point sampling
    radii = np.sqrt(2) * (
        np.linspace(0, (img.shape[2] / 2), numRadii, endpoint=False, retstep=False)
        + 0.5
    )
    theta = np.linspace(initialAngle, finalAngle, numAngles, endpoint=False)

    # convert to tensors to accelerate point sampling
    r, theta = torch.meshgrid(
        torch.tensor(radii, device=img.device),
        torch.tensor(theta, device=img.device),
        indexing="xy",
    )
    center = torch.tensor(center, device=img.device)
    xCartesian, yCartesian = getCartesianPointsTorch(r, theta, center)

    pad = 3
    if border == "constant":
        # Pad image by 3 pixels and then offset all of the desired coordinates by 3
        img = fun.pad(img, (pad, pad, pad, pad), "replicate")
        xCartesian += pad
        yCartesian += pad

    gx = 2.0 * (xCartesian / (img.shape[3] - 1)) - 1.0
    gy = 2.0 * (yCartesian / (img.shape[2] - 1)) - 1.0
    desiredCoords = torch.stack((gx, gy), 2)
    desiredCoords = desiredCoords.unsqueeze(0).repeat(
        img.shape[0], *(1,) * len(desiredCoords.shape)
    )

    polarImage = fun.grid_sample(
        img.float(),
        desiredCoords.float(),
        padding_mode="zeros",
        mode=INTERP_TYPE,
        align_corners=True,
    )

    return polarImage


def batchpolar2img(
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
    Converts batch of polar images to cartesian image.
    Expects multichan images, but chan & batch dimension can be 1.

    Parameters
    ----------
    img : torch.Tensor
        Polar image to be converted. Should be (B, C, A, R) where A is angles and R is radii.

    imageSize : tuple, optional
        Size of the image to be returned. If None, the image is assumed to be (B, C, R, R).

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
    imageSize = (img.shape[-1], img.shape[-1])
    if center is None:
        center = ((imageSize[-2] - 1) / 2.0, (imageSize[-1] - 1) / 2.0)

    initialAngle = np.pi / 4
    finalAngle = 2 * np.pi + np.pi / 4

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
    xs = torch.arange(0, imageSize[-1], device=img.device)
    ys = torch.arange(0, imageSize[-2], device=img.device)
    x, y = torch.meshgrid(xs, ys, indexing="xy")

    # Take cartesian grid and convert to polar coordinates
    r, theta = getPolarPointsTorch(x, y, center)

    # Offset the radius by the initial source radius
    r = r - initialRadius

    # Offset the theta angle by the initial source angle
    # The theta values may go past 2pi, so they are modulo 2pi.
    # Note: This assumes initial source angle is positive
    theta = torch.remainder(theta - initialAngle + 2 * np.pi, 2 * np.pi)

    # Scale the radius using scale factor
    r = r * scaleRadius

    # Scale the angle from radians to pixels using scale factor
    theta = theta * scaleAngle

    pad = 3

    if border == "constant":
        # Pad image by 3 pixels and then offset all of the desired coordinates by 3
        img = fun.pad(img, (pad, pad, pad, pad), "replicate")
        r += pad
        theta += pad

    gr = 2.0 * (r / (img.shape[3] - 1)) - 1.0
    gtheta = 2.0 * (theta / (img.shape[2] - 1)) - 1.0
    desiredCoords = torch.stack((gr, gtheta), 2)
    desiredCoords = desiredCoords.unsqueeze(0).repeat(
        img.shape[0], *(1,) * len(desiredCoords.shape)
    )

    cartImage = fun.grid_sample(
        img.float(),
        desiredCoords.float(),
        padding_mode="zeros",
        mode=INTERP_TYPE,
        align_corners=True,
    )

    return cartImage
