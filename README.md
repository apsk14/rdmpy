# rdmpy: Spatially-varying deblurring in Python.

[![Paper](https://img.shields.io/badge/paper-%09arXiv%3A2003.12673-yellow.svg)](https://arxiv.org/abs/2206.08928)

## UPDATES:
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
  * Official PyPI package to follow shortly.
  * Includes a new deep deblurring model, DeepRD (try it out with our pretrained models).
  * Please contact me if you have any issues or have ideas for the package!


## Background
Official implementations of Seidel fitting, ring convolution, ring deconvolution, sheet convolution, and sheet deconvolution. 

Ring convolution allows one to compute the forward model of any rotationally symmetric imaging system. Simiarly, Ring deconvolution offers a fast and more accurate alternative to deconvolution (e.g., Wiener filter, Richardson-Lucy) by accounting for radially-varying blur. Both only require a single calibration image of a few randomly scattered PSFs. They are implemented with the two functions ```ring_convolve``` and ```ring_deconvolve```, respectively. Sheet convolution/deconvolution is the same for systems with lateral symmetry, like light-sheet microscopes.

This repository is still in early stages and will constantly be updated with new functionality and bug-fixes. In light of this **please run a ```git pull``` before every use.** It will soon be a PyPI package directly installable with pip.


## Installation

Full installation should take no more than 10 minutes.

To get the code, clone this repo with ```git clone https://github.com/apsk14/rdmpy.git```.

Next, we need to manage all the necessary packages.

The original code was developed in Python 3.8.5 with fairly standard packages (listed below). We believe that the code will run with an environment that already has most of these---just add on the missing ones. 

Main packages:
* numpy 1.20.2
* pytorch 1.8.1
* scipy 1.6.2
* scikit-image 0.17.2
* pillow 8.2.0
* matplotlib 3.2.2
* tqdm 4.65.0
* kornia 0.5.3 
* jupyter 1.0.0

If you would like to install from scratch, please follow these steps using Anaconda:
1) If you don't already have Anaconda or another package manager installed, we recommend [Miniconda](https://docs.anaconda.com/free/miniconda/miniconda-install/),
2) Create a new environment ```conda create -n "rdm" python=3.8.5```
3) Install pytorch first! This is important as it will make the rest smoother and will mean less total installs. ```conda install pytorch=1.8.1 pytorch-cuda=11.8 -c pytorch -c nvidia```. If you plan to only use cpu (not recommended), you can drop the pytorch-cuda part.
4) Next, conda install scipy=1.6.2, scikit-image (this may be slow), matplotlib=3.2.2, tqdm, and jupyter.
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

