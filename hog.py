import cv2
import os
import matplotlib.pyplot as plt
import skimage
import numpy as np

from sklearn.linear_model import SGDClassifier
from sklearn.utils import shuffle
from skimage import feature
from skimage import exposure

from sklearn import svm

# image = cv2.imread('images/hog.jpg')
# image = cv2.resize(image, (200, 200))
# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
# (H, hog_image) = feature.hog(image, orientations=9,
#                              pixels_per_cell=(8, 8), cells_per_block=(2, 2),
#                              block_norm='L1', visualize=True, transform_sqrt=True)
#
# hog_image = skimage.exposure.rescale_intensity(hog_image, out_range=(0, 255))
# hog_image = hog_image.astype("uint8")

mapping = {}
images = []
labels = []

for i, brand in enumerate(os.listdir("dataset_hog")):
    if brand.startswith('.'):
        continue
    mapping[i] = brand
    brand_directory = os.path.join("dataset_hog", brand)
    for filename in os.listdir(brand_directory):
        if filename.startswith('.'):
            continue

        image = cv2.imread(os.path.join(brand_directory, filename))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (200, 200))
        images.append(image)
        labels.append(i)

images, labels = shuffle(images, labels)

features = [feature.hog(image, orientations=9,
                        pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                        block_norm='L1', visualize=True, transform_sqrt=True)[0] for image in images]

sgd = SGDClassifier()
sgd.fit(features, labels)

svm = svm.SVC(decision_function_shape='ovr')
svm.fit(features, labels)

# for filename in os.listdir("dataset_hog_test"):
# image = cv2.imread(os.path.join("images/yellow.jpg", filename))
# image = cv2.imread("images/yellow.jpg")
# image = cv2.imread("images/bordo.jpg")
image = cv2.imread("images/006.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = cv2.resize(image, (200, 200))

(H, hog_image) = feature.hog(image, orientations=9,
                             pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                             block_norm='L1', visualize=True, transform_sqrt=True)

result = svm.predict([H])[0]
print(result)

fd = H.reshape(1, -1)
score = svm.decision_function([H])
print(score)

cv2.putText(image, mapping[result], (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
plt.imshow(image)
plt.show()
