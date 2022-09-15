#Code implementing the user functions for lri-deblur. These functions can be directly imported via from lri-deblur import calibrate, blur, deblur
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import torch
import numpy as np
import torch.fft as fft
from skimage import io
from skimage.transform import resize
from _src import opt, seidel, lri_forward, util, polar_transform
from functions import *
import pdb

if torch.cuda.is_available():
    device = torch.device('cuda:1')
else:
    device = torch.device('cpu')
    
print('Using ' +str(device) + ' for computation')

# first, a test image and pick a dimension
dim= 512
num_psfs = 512

# object
#obj1, point_list = util.getRadialImpulse(dim=[dim,dim], radii= [0, 50, 100, 150, 200], sparsity=4) # [0, 20, 40, 60, 80, 100, 120, 140, 160, 180]
obj1 = torch.tensor(resize(io.imread('test_images/baboon.png'), [dim, dim])).float()
obj_pol1 = polar_transform.img2polar(obj1, numRadii=num_psfs)
obj_rec1 = polar_transform.polar2img(obj_pol1, obj1.shape)

print(torch.norm(obj1 - obj_rec1))

util.show((obj1 - obj_rec1).abs())
pdb.set_trace()




obj2, point_list = util.getRadialImpulse(dim=[dim,dim], radii= [0, 50, 100, 150, 200], sparsity=4) # [0, 20, 40, 60, 80, 100, 120, 140, 160, 180]
obj1 = torch.tensor(obj1).float()
obj2 = torch.tensor(obj2).float()
#obj = obj/obj.sum()

obj_pol1 = polar_transform.img2polar(obj1, numRadii=num_psfs)
obj_pol2 = polar_transform.img2polar(obj2, numRadii=num_psfs)
obj_pol_sum = obj_pol1 * obj_pol2

obj_rec1 = polar_transform.polar2img(obj_pol1, obj1.shape)
obj_rec2 = polar_transform.polar2img(obj_pol2, obj2.shape)
obj_rec_sum = polar_transform.polar2img(obj_pol_sum, obj2.shape)
# obj_rec[obj_rec<0] = 0


print(torch.norm(obj1 - obj_rec1))
print(torch.norm(obj2 - obj_rec2))

print(torch.norm(obj1+obj2 - obj_rec_sum))

# print(obj.max())
# print(obj_rec.max())

# print(obj.sum())
# print(obj_rec.sum())

pdb.set_trace()
#util.show(obj_pol)
util.show(torch.cat((obj1*obj2,obj_rec_sum), dim=1))
#util.show((obj-obj_rec).abs())



# first let's really test the polar





