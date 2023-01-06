import cv2
import os
import matplotlib.pyplot as plt
import skimage
import joblib
import numpy as np

from sklearn.linear_model import SGDClassifier
from sklearn.utils import shuffle
from skimage import feature
from skimage import exposure

from sklearn import svm

classes = {}
images = []
labels = []

for i, brand in enumerate(os.listdir("dataset_hog")):
    if brand.startswith('.'):
        continue
    classes[i] = brand
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

svm = svm.SVC(decision_function_shape='ovr', probability=True)
svm.fit(features, labels)
joblib.dump(svm, 'model/svm/svm.sav')

# for filename in os.listdir("dataset_hog_test"):
# image = cv2.imread(os.path.join("images/yellow.jpg", filename))

# image = image_test = cv2.imread("dataset_hog_test/yellow.jpg")
image = image_test = cv2.imread("dataset_hog_test/bordo.jpg")
# image = image_test = cv2.imread("dataset_hog_test/fake_3.jpg")

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

score = svm.predict_proba([H])
print(score)

cv2.putText(image, classes[result], (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
plt.imshow(image)
plt.show()

cv2.putText(image_test, classes[result], (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
plt.imshow(cv2.cvtColor(image_test, cv2.COLOR_BGR2RGB))
plt.show()
