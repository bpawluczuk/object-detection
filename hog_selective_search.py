import cv2
import os
import matplotlib.pyplot as plt
import skimage
import random
from sklearn.linear_model import SGDClassifier
from sklearn.utils import shuffle
from skimage import feature
from skimage import exposure
from sklearn import svm

image = cv2.imread("images/shower_all.jpg")

image = image.copy()
image = cv2.resize(image, (600, 800), interpolation=cv2.INTER_AREA)
# cv2.imshow("Output", image)
# key = cv2.waitKey(0) & 0xFF

ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
ss.setBaseImage(image)
# ss.switchToSelectiveSearchQuality()
ss.switchToSelectiveSearchFast()
rects = ss.process()

(H, W) = image.shape[:2]
len_rects = len(rects)

output = image.copy()

model = SGDClassifier()
modelSVC = svm.SVC()

inc = 0
for (x, y, w, h) in rects:

    if w / float(W) < 0.05 or w / float(W) > 0.1 or h / float(H) < 0.05 or h / float(H) > 0.2:
        continue

    inc = inc + 1
    color = [random.randint(0, 255) for j in range(0, 3)]
    cv2.rectangle(output, (x, y), (x + w, y + h), color, 1)

    roi = image[y:y + h, x:x + w]
    # roi = cv2.resize(roi, (200, 200))
    # cv2.imwrite("output/" + str(inc) + "_000.jpg", roi)

    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi = cv2.resize(roi, (200, 200))

    (Ha, hog_image) = feature.hog(roi, orientations=9,
                                 pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                                 block_norm='L1', visualize=True, transform_sqrt=True)

    result = model.predict([H])[0]

print(inc)
cv2.imshow("Output", output)
key = cv2.waitKey(0) & 0xFF
