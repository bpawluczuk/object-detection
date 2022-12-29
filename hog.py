import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import skimage

from sklearn.linear_model import SGDClassifier
from sklearn.utils import shuffle
from skimage import feature
from skimage import exposure

# hog = cv2.HOGDescriptor()
# im = cv2.imread(sample)
# h = hog.compute(im)

image = cv2.imread('images/hog.jpg')
image = cv2.resize(image, (200, 200))
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

(hog, hog_image) = feature.hog(image, orientations=9,
                               pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                               block_norm='L1', visualize=True, transform_sqrt=True)

hog_image = skimage.exposure.rescale_intensity(hog_image, out_range=(0, 255))
hog_image = hog_image.astype("uint8")
plt.imshow(hog_image, cmap='gray')
plt.show()

# cv2.imshow('HOG Image', hog_image)
# cv2.imwrite('hog_flower.jpg', hog_image * 255.)
# cv2.waitKey(0)
