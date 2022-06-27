#file for demonstrating a run of LRI
from functions import *
import pdb
from skimage import io
from skimage.transform import resize
from PIL import Image


if torch.cuda.is_available():
    device = torch.device('cuda:1')
else:
    device = torch.device('cpu')



def main():

    # pick dimension of all images and PSFs
    dim = 512

    #pick a calibration image or some random seidel coefficients 
    #calib_image=None
    #seidel_coeffs =  [0, 3, 2, 1.5, 0, 0]
    #seidel_coeffs =  [0, 0.01, 0, 0, 0, 0]
    calib_image = np.array(Image.open('test_images/calibration_image.tif')) 
    seidel_coeffs=None


    #get a radial stack of the system's PSFs (either provide seidel coefficients or a calibration image with a few PSFs in it)
    psf_stack_roft, seidel_coeffs = calibrate(calib_image, dim, dim, seidel_coeffs=seidel_coeffs, device=device)

    # read in a test object
    obj = resize(io.imread("test_images/baboon.png", as_gray=True), [dim,dim])

    # blur using the LRI forward model
    blurred = blur(obj, psf_stack_roft, device=device)

    util.show(blurred.cpu().detach())

    # deblur! 
    recon = deblur(blurred, psf_stack_roft, device=device, artifact_correction=True)

    util.show(recon)





if __name__ == '__main__':
    main()