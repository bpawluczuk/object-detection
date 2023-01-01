import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import skimage
import pickle

from sklearn.linear_model import SGDClassifier
from sklearn.utils import shuffle
from skimage import feature
from skimage import exposure

from sklearn import svm
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

image = cv2.imread('images/hog.jpg')
image = cv2.resize(image, (200, 200))
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

(H, hog_image) = feature.hog(image, orientations=9,
                             pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                             block_norm='L1', visualize=True, transform_sqrt=True)

# print(len(H))

hog_image = skimage.exposure.rescale_intensity(hog_image, out_range=(0, 255))
hog_image = hog_image.astype("uint8")
# plt.imshow(hog_image, cmap='gray')
# plt.show()

mapping = {}
images = []
labels = []

for i, brand in enumerate(os.listdir("dataset_hog")):
    if brand.startswith('.'): continue
    mapping[i] = brand
    brand_directory = os.path.join("dataset_hog", brand)
    for filename in os.listdir(brand_directory):
        if filename.startswith('.'): continue

        image = cv2.imread(os.path.join(brand_directory, filename))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (200, 200))
        images.append(image)
        labels.append(i)

images, labels = shuffle(images, labels)

features = [feature.hog(image, orientations=9,
                        pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                        block_norm='L1', visualize=True, transform_sqrt=True)[0] for image in images]

# print(len(features[1]))
# print(len(features))

model = SGDClassifier()
model.fit(features, labels)

modelSVC = svm.SVC()
modelSVC.fit(features, labels)

# print(labels)
# print(mapping)
# print(images)

for filename in os.listdir("dataset_hog_test"):
    image = cv2.imread(os.path.join("dataset_hog_test", filename))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (200, 200))

    (H, hog_image) = feature.hog(image, orientations=9,
                                 pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                                 block_norm='L1', visualize=True, transform_sqrt=True)

    result = model.predict([H])[0]

    fd = H.reshape(1, -1)
    score = model.decision_function([H])
    print(score)

    cv2.putText(image, mapping[result], (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    plt.imshow(image)
    plt.show()
