import cv2
import os
import matplotlib.pyplot as plt
import skimage
import joblib
import random
import numpy as np
from skimage import feature

svm = joblib.load('model/svm/svm.sav')
classes = ["001", "002", "003", "004", "005"]

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

inc_total = 0
inc_pred = 0
inc = 0

for (x, y, w, h) in rects:

    inc_total = inc_total + 1

    color = [random.randint(0, 255) for j in range(0, 3)]
    cv2.rectangle(output, (x, y), (x + w, y + h), color, 1)

    roi = image[y:y + h, x:x + w]
    image_out = roi.copy()
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi = cv2.resize(roi, (200, 200))

    (H, hog_image) = feature.hog(roi, orientations=9,
                                 pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                                 block_norm='L1', visualize=True, transform_sqrt=True)

    result = svm.predict([H])[0]
    score = svm.predict_proba([H])
    print(score)

    if classes[result] == "001" and (100 * np.max(score)) >= 0.9:
        inc_pred = inc_pred + 1
        print(
            "This image most likely belongs to {} with a {:.2f} percent confidence."
            .format(classes[result], 100 * np.max(score))
        )
        color = [255, 128, 128]
        cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)

    if classes[result] == "002" and (100 * np.max(score)) >= 0.9:
        inc_pred = inc_pred + 1
        print(
            "This image most likely belongs to {} with a {:.2f} percent confidence."
            .format(classes[result], 100 * np.max(score))
        )
        color = [128, 255, 128]
        cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)

    # inc = inc + 1
    # cv2.imwrite("garbage/" + str(inc) + "_g.jpg", image_out)

print("Total: ", inc_total)
print("Predict: ", inc_pred)

cv2.imshow("Output", output)
key = cv2.waitKey(0) & 0xFF
