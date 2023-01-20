import tensorflow as tf
import random
import cv2
import numpy as np
import seaborn as sns
import time
import matplotlib.pyplot as plt
from tensorflow import keras

from Canvas import Canvas

sns.set()
startTime = time.time()

# ===============================================================================================
img_height = 512
img_width = 512
channels = 3
CANVAS_SHAPE = (img_height, img_width, channels)

canvas = Canvas(CANVAS_SHAPE)

model = keras.models.load_model('model/vgg')
print("Load model...")
# ===============================================================================================

images_predicted = []
score_predicted = []
boxes = []

class_names = ["001", "002"]
image = cv2.imread("images/test_shape.jpg")

# Scale down
p = 0.30
w = int(image.shape[1] * p)
h = int(image.shape[0] * p)
image = cv2.resize(image, (w, h))

print("Search...")

ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
ss.setBaseImage(image)
ss.switchToSelectiveSearchFast()
# ss.switchToSelectiveSearchQuality()
rects = ss.process()

(H, W) = image.shape[:2]
len_rects = len(rects)

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
output = image.copy()


# plt.axis('off')
# plt.imshow(output)
# plt.show()


def get_middle_point(x_pos, y_pos, w_len, h_len):
    return int(x_pos + w_len / 2), int(y_pos + h_len / 2)


def bounding_boxes_merge(bounding_boxes, offset_x=0, offset_y=0):
    results = []
    for box in enumerate(bounding_boxes):
        x_box, y_box, w_box, h_box = box[1]
        xm, ym = get_middle_point(x_box, y_box, w_box, h_box)
        inside = False
        for result in enumerate(results):
            x_res, y_res, w_res, h_res = result[1]
            xmr, ymr = get_middle_point(x_res, y_res, w_res, h_res)
            if xmr - offset_x < xm < xmr + offset_x and ymr - offset_y < ym < ymr + offset_y:
                inside = True
        if not inside:
            results.append(box[1])

    return results


inc_total = 0
for (x, y, w, h) in rects:

    inc_total = inc_total + 1

    if w > 200 * p or w < 160 * p or h > 450 * p or h < 360 * p:
        continue

    roi = image[y:y + h, x:x + w]
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

    boxes.append((x, y, w, h))

    # color = [random.randint(0, 255) for j in range(0, 3)]
    # color = (0, 255, 0)
    # cv2.rectangle(output, (x, y), (x + w, y + h), color, 1)

    # xm, ym = get_middle_point(x, y, w, h)
    # output = cv2.circle(output, (xm, ym), radius=2, color=(0, 0, 255), thickness=-1)

boxes_m = bounding_boxes_merge(boxes, 25, 55)
# boxes_m = boxes

for (x, y, w, h) in boxes_m:
    color = [random.randint(0, 255) for j in range(0, 3)]
    color = (0, 255, 0)
    cv2.rectangle(output, (x, y), (x + w, y + h), color, 1)

plt.axis('off')
plt.imshow(output)
plt.show()
