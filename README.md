# rdmpy: Spatially-varying deblurring in Python.

[![Paper](https://img.shields.io/badge/paper-%09arXiv%3A2003.12673-yellow.svg)](https://arxiv.org/abs/2206.08928)

## UPDATES:
* [August 30th 2023] Introducing RDMPY v1!
  * Official package to follow shortly.
  * Includes a new deep deblurring model, DeepRD (try it out with our pretrained models).
  * Please contact me if you have any issues or have ideas for the package!


## Background
Official implementations of Seidel fitting, ring convolution, and ring deconvolution. 

Ring convolution allows one to compute the forward model of any rotationally symmetric imaging system. Simiarly, Ring deconvolution offers a fast and more accurate alternative to deconvolution (e.g., Wiener filter, Richardson-Lucy) by accounting for radially-varying blur. Both only require a single calibration image of a few randomly scattered PSFs. they implemented with the two functions ```ring_convolve``` and ```ring_deconvolve```, respectively.

This repository is still in early stages and will constantly be updated with new functionality and bug-fixes. In light of this **please run a ```git pull``` before every use.** It will soon be a PyPI package directly installable with pip.

## Useage

For a quick walkthrough of the basic utility of this package please see ```example.ipynb```. 

All functions for rdmpy can be found in either ```calibrate.py```, ```blur.py```, or ```deblur.py```. Each function is fully documented within those files.

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

