import cv2
import os
import matplotlib.pyplot as plt
import skimage
import joblib
import numpy as np

from skimage import feature
from sklearn import svm

svm = joblib.load('model/svm/svm.sav')
classes = ["000", "001", "002", "003", "004", "005"]

# image = image_test = cv2.imread("dataset_hog_test/yellow.jpg")
image = image_test = cv2.imread("dataset_hog_test/bordo.jpg")
# image = image_test = cv2.imread("dataset_hog_test/fake_3.jpg")

image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = cv2.resize(image, (200, 200))

(H, hog_image) = feature.hog(image, orientations=9,
                             pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                             block_norm='L1', visualize=True, transform_sqrt=True)

result = svm.predict([H])[0]
score = svm.predict_proba([H])
print(score)

cv2.putText(image_test, classes[result], (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
plt.imshow(cv2.cvtColor(image_test, cv2.COLOR_BGR2RGB))
plt.show()
