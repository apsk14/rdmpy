#file for demonstrating a run of LRI
from functions import *
import pdb
from skimage import io
from skimage.transform import resize





def main():

    if torch.cuda.is_available():
        device = torch.device('cuda:1')
    else:
        device = torch.device('cpu')

    dim = 384
    seidel_coeffs =  [0, -0.9105, 0.0549, -0.6531, 0, 0]
    psf_stack_roft, seidel_coeffs = calibrate(None, dim, dim, seidel_coeffs=seidel_coeffs, device=device)

    obj = resize(io.imread("baboon.png", as_gray=True), [dim,dim])
    blurred = blur(obj, psf_stack_roft, device=device)

   # blurry = resize(io.imread("blurry_baboon.png", as_gray=True), [dim, dim])
    recon = deblur(blurred, psf_stack_roft, device=device)

    pdb.set_trace()




if __name__ == '__main__':
    main()