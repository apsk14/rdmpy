# rdmpy: Spatially-varying deblurring in Python.

[![Static Badge](https://img.shields.io/badge/paper-Nature%20Methods-yellow)](https://www.nature.com/articles/s41592-025-02684-5)

## Announcement
The official rdmpy python package will be released in soon on PyPI and will be pip installable. If you have any feature suggestions or ideas please don't hesitate to reach out.


## UPDATES:
* [May 2025]
  * We can now do patchwise ring deconvolution! This allows for much larger images (upward of 2048x2048) to be ring   deconvolved faster with less memory useage. If you want to increase speed and decrease memory useage just increase the patch_size argument.
  * To try this new feature, check out the large_images.ipynb notebook. We are able to ring deconvolve much larger image crops than in our paper.
  * Patchwise sheet deconvolution will be coming shortly!

* [April 2025] MAJOR UPDATE!
  * Our paper is officially published in [Nature Methods](https://www.nature.com/articles/s41592-025-02684-5).
  * The codebase has been updated to accomodate the latest dependency updates. See the updated package versioning below.
  * RDMPY will become pip installable soon!
  * Please don't hesistate to reach out with any bug finds or desired features. We will continue to create new releases every few months
* [November 2024] MAJOR UPDATE!
  * Added Sheet convolution/deconvolution for deblurring images from light-sheet microscopes. Check out experiments/light-sheet.ipynb.
  * Added higher-order Seidel coefficients for systems that have higher-order aberrations.
  * Note: these updates require upgrading pytorch to 2.4.1.
* [March 11th 2024] New Demos with experimental data
  * Check out demo.ipynb and mmf.ipynb for easy-to-run examples of rdmpy
  * If you have some bigger GPUs, try out the miniscope.ipynb or multicolor.ipynb
* [December 4th 2023] Accelerated implementations of ring deconvolution.
  * Now run deconvolution up to 15x faster!
  * Run batched ring convolution up to 100x faster!
* [August 30th 2023] Introducing RDMPY v1!
  * Includes a new deep deblurring model, DeepRD (try it out with our pretrained models).
  * Please contact me if you have any issues or have ideas for the package!


## Background
Official implementations of Seidel fitting, ring convolution, ring deconvolution, sheet convolution, and sheet deconvolution. 

Ring convolution allows one to compute the forward model of any rotationally symmetric imaging system. Simiarly, Ring deconvolution offers a fast and more accurate alternative to deconvolution (e.g., Wiener filter, Richardson-Lucy) by accounting for radially-varying blur. Both only require a single calibration image of a few randomly scattered PSFs. They are implemented with the two functions ```ring_convolve``` and ```ring_deconvolve```, respectively. Sheet convolution/deconvolution is the same for systems with lateral symmetry, like light-sheet microscopes.

This repository is still in early stages and will constantly be updated with new functionality and bug-fixes. In light of this **please run a ```git pull``` before every use.** It will soon be a PyPI package directly installable with pip.


## Installation

Instructions for pip installation will be here soon!

Full installation from scratch should take no more than 10 minutes.

To get the code, clone this repo with ```git clone https://github.com/apsk14/rdmpy.git```.

Next, we need to manage all the necessary packages.

The original code was developed in Python 3.8.5 with fairly standard packages (listed below). We believe that the code will run with an environment that already has most of these---just add on the missing ones. 

Main packages:
* python 3.12.2
* numpy 1.26.4
* pytorch 2.4.1
* scipy 1.13.1
* scikit-image 0.24.0
* pillow 10.4.0
* matplotlib 3.9.2
* tqdm 4.66.5
* kornia 0.7.1
* jupyter 1.0.0

If you would like to install from scratch, please follow these steps using Anaconda:
1) If you don't already have Anaconda or another package manager installed, we recommend [Miniconda](https://docs.anaconda.com/free/miniconda/miniconda-install/) or [Micromamba](https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html). Micromamba can work better at resolving dependencies if you are having issues with Miniconda.
2) Create a new environment ```conda create -n "rdm" python=3.12.2```
3) Install pytorch first! This is important as it will make the rest smoother and will mean less total installs. ```conda install pytorch=2.4.1 pytorch-cuda=11.8 -c pytorch -c nvidia```. If you plan to only use cpu (not recommended), you can drop the pytorch-cuda part. If you will use GPU please select the cuda version according to your hardware.
4) Next, conda install scipy=1.13.1, scikit-image (this may be slow), matplotlib=3.9.2, tqdm, and jupyter.
5) Finally we will use conda-forge for kornia and install it as ```conda install conda-forge::kornia```. This step can also take some time.

After that, we are ready to run some rdmpy! You can immediately try out demo.ipynb to see if everything works out. If you don't have enough GPU memory and/or the process is slow on your CPU, feel free to reduce the dim parameter in the beginning of the notebook.

## Useage

Disclaimer: All jupyter notebooks can also just be viewed if the reader cannot or does not want to run them. 

For a walkthrough and instructions of the basic utility of this package please see ```demo.ipynb```. Depending on the notebook and computational resources available, this notebook should take no more than 10-15 minutes to run. On a linux server with a single Nvidia RTX A6000 GPU, it will completely run in a few minutes. 

For more advanced useage, navigate to experiments/ and run ```bash get_data.sh experiment_name``` where experiment_name can be either mmf, miniscope, multicolor, or light-sheet. For example, ```bash get_data.sh mmf```. The open up the corresponding jupyter notebook and have fun! MMF is quite small and runs easily on any system. The other 2 require large gpus (>25GB Ram) to operate quickly at full scale. Moreover, they will require ~8GB of storage for the data. 

Comprehensive documentation for rdmpy can be found in ```calibrate.py```, ```blur.py```, and ```deblur.py```. Each function is fully documented within those files under the function headers. 

A full documentation webpage is coming soon!

## Acknowledgements

We used code inspired by: 

Computational Fourier Optics: A MATLAB Tutorial by David Voelz

https://github.com/addisonElliott/polarTransform/tree/master

https://github.com/MicroscPSF/MicroscPSF-Py?tab=readme-ov-file


## Contact
Please reach out to apkohli@berkeley.edu with any questions or concerns!

