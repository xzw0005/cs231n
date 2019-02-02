'''
Created on Jan 27, 2019

@author: wangxing
'''
import numpy as np
import matplotlib.pyplot as plt
from a2.utils import get_CIFAR10_data, eval_numerical_gradient_array, eval_numerical_gradient, rel_error
from a2.layers import *
from a2.solver import Solver

# data = get_CIFAR10_data()
# for k, v in data.items():
#     print('%s: '%k, v.shape)

from scipy.misc import imread, imresize
kitten, puppy = imread('kitten.jpg'), imread('puppy.jpg')
print(kitten.shape, puppy.shape)
d = kitten.shape[1] - kitten.shape[0]
kitten_cropped = kitten[:, d//2:-d//2, :]
img_size = 200
x = np.zeros((2, 3, img_size, img_size))
x[0,:,:,:] = imresize(puppy, (img_size, img_size)).transpose((2,0,1))
x[1,:,:,:] = imresize(kitten_cropped, (img_size, img_size)).transpose((2,0,1))

# Set up 2 filters, each 3x3
w = np.zeros((2, 3, 3, 3)) 
# The 1st filter converts image to grayscale
w[0, 0, :, :] = [[0,0,0], [0, 0.3, 0], [0,0,0]] # red channel filter
w[0, 1, :, :] = [[0,0,0], [0, 0.6, 0], [0,0,0]] # greed channel filter
w[0, 2, :, :] = [[0,0,0], [0, 0.1, 0], [0,0,0]] # blue channel filter
# 2nd filter detects horizontal edges in the blue channel
w[1, 2, :, :] = [[1, 2, 1], [0,0,0], [-1, -2, -1]]

b = np.array([0, 128])
out, _ = conv_forward_naive(x, w, b, conv_param={'stride':1, 'pad':1})

def imshow_noax(img, normalize=True):
    if normalize:
        img_max, img_min = np.max(img), np.min(img)
        img = 255. * (img - img_min) / (img_max - img_min)
    plt.imshow(img.astype('uint8'))
    plt.gca().axis('off')
plt.subplot(2, 3, 1)
imshow_noax(puppy, normalize=False)
plt.title('Original Image')
plt.subplot(2, 3, 2)
imshow_noax(out[0, 0])
plt.title('Grayscale')
plt.subplot(2, 3, 3)
imshow_noax(out[0, 1])
plt.title('Edges')
plt.subplot(2, 3, 4)
imshow_noax(kitten_cropped, normalize=False)
plt.subplot(2, 3, 5)
imshow_noax(out[1, 0])
plt.subplot(2, 3, 6)
imshow_noax(out[1, 1])
plt.show()



























