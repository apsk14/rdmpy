#Welcome to the official repository for rdmpy, a python package for spatially-varying deblurring. Background info can be found in the paper below:

[![Paper](https://img.shields.io/badge/paper-%09arXiv%3A2003.12673-yellow.svg)](https://arxiv.org/abs/2206.08928)

## UPDATES:
* [August 30th 2023] Introducing to RDMPY v1!
  * Official package to follow shortly.
  * Includes a new deep deblurring model, DeepRD (try it out with our pretrained models).
  * Please contact me if you have any issues or have ideas for the package!


## Background
Official implementations of Seidel fitting, ring convolution, and ring deconvolution. 

Ring convolution allows one to compute the forward model of any rotationally symmetric imaging system. Simiarly, Ring deconvolution offers a fast and more accurate alternative to deconvolution (e.g., Wiener filter, Richardson-Lucy) by accounting for radially-varying blur. Both only require a single calibration image of a few randomly scattered PSFs. they implemented with the two functions ```blur``` and ```deblur```, respectively.

This repository is still in early stages and will constantly be updated with new functionality and bug-fixes. In light of this **please run a ```git pull``` before every use.** It will soon be a PyPI package directly installable with pip.

## Useage

For a quick example of the basic utility of this package please see ```simple_example.ipynb```. For a more in-depth overview and some real-life data see ```example.ipynb```. For full documentation of each function see ```functions.py```.

There are 4 main functions of interest in this package. They can all be found in ```functions.py```. Each function can be run using the LRI (linear revolution invariant) or LSI (linear shift-invariant) assumption. This is specified by the ```model``` parameter which is either 'lri' or 'lsi'

**1) blind_deblur**

If no PSFs are available, one can simply pass their blurry image into blind_deblur to get a deblurred estimate. 
```
deblurred_image, psf_estimate, seidel_coeffs = blind_deblur(blurred_image)
```
and that's it!!

**2) calibrate**

Here we pass in either a calibration image such as ```test_images/calibration_image.tif``` or Seidel aberration coefficients (i.e., [sphere, astigmatism, coma, field curvature, distortion]). We get back a stack of the rotational fourier transforms (see paper appendix E) of PSFs along a radial line and the Seidel coefficients.
```
seidel_coeffs, psf_stack = calibrate(calib_image, model, dim, seidel_coeffs=None)
```
**3a) blur**

With PSFs in hand we can simulate the blurring of an imaging system using the LRI forward model. We simply pass in a test object such as ```test_images/baboon.png``` and our PSF stack obtained from the calibration step.
```
blurred_image = blur(object, psf_stack, model)
```

**3b) deblur**

Alternatively we can choose to invert the LRI forward model to instead perform image deblurring. Again we only need the PSF stack from step 1 in addition to the blurry image we would like to deblur.
```
deblurred_image = deblur(blurred_image, psf_stack, model)
```


The necessary packages to run this code are:
* numpy
* pytorch
* scipy
* skimage
* PIL
* cv2
* matplotlib
* tqdm


## Contact
Please reach out to apkohli@berkeley.edu with any questions or concerns!

