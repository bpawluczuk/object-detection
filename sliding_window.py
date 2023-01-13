import numpy as np
from imutils.object_detection import non_max_suppression
import imutils
from joblib.numpy_pickle_utils import xrange
from skimage.feature import hog
import cv2
from skimage import color
import matplotlib.pyplot as plt
import os
import glob
import sys
from skimage.transform import pyramid_gaussian


def sliding_window(image, window_size, step_size):
    '''
    This function returns a patch of the input 'image' of size
    equal to 'window_size'. The first image returned top-left
    co-ordinate (0, 0) and are increment in both x and y directions
    by the 'step_size' supplied.
    So, the input parameters are-
    image - Input image
    window_size - Size of Sliding Window
    step_size - incremented Size of Window
    The function returns a tuple -
    (x, y, im_window)
    '''
    for y in xrange(0, image.shape[0], step_size[1]):
        for x in xrange(0, image.shape[1], step_size[0]):
            yield x, y, image[y: y + window_size[1], x: x + window_size[0]]


image = cv2.imread("images/shower_test.jpg")
print("oryginal: ", image.shape[0:2])
image = imutils.resize(image, width=min(600, image.shape[1]))
min_wdw_sz = (64, 128)
step_size = (64, 128)
downscale = 1.6

print("resize: ", image.shape[0:2])

for (x, y, im_window) in sliding_window(image, min_wdw_sz, step_size):
    print(im_window.shape[0:2])
    cv2.imshow("Output", im_window)
    key = cv2.waitKey(0) & 0xFF

print("end")
