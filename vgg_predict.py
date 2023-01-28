import tensorflow as tf
import random
import cv2
import numpy as np
import seaborn as sns
import time
import matplotlib.pyplot as plt
from tensorflow import keras

from BoundingBox import merge_boxes, sort_boxes
from Canvas import Canvas

sns.set()
startTime = time.time()

# ============================= Classificator ============================================

img_height = 512
img_width = 512
channels = 3

CANVAS_SHAPE = (img_height, img_width, channels)
canvas = Canvas(CANVAS_SHAPE)

model = keras.models.load_model('model/vgg')
print("Load model...")

# ======================================= Input image ===================================

images_predicted = []
score_predicted = []
boxes = []

class_names = ["001", "002"]

image = cv2.imread("images/test_shape.jpg")

# Scale down
percent_of_size = 0.40
w = int(image.shape[1] * percent_of_size)
h = int(image.shape[0] * percent_of_size)
image = cv2.resize(image, (w, h))

# ============================ Object dimensions ========================================

# object_w = int(500)
# object_h = int(1000)
# offset_w = int(40)
# offset_h = int(200)

object_w = int(200)
object_h = int(460)
offset_w = int(60)
offset_h = int(60)

object_w = int(object_w * percent_of_size)
object_h = int(object_h * percent_of_size)
offset_w = int(offset_w * percent_of_size)
offset_h = int(offset_h * percent_of_size)

box_offset_w = int(object_w / 2)
box_offset_h = int(object_h / 2)

# ============================== Search Region ==========================================

print("Search...")

ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
ss.setBaseImage(image)
# ss.switchToSelectiveSearchFast()
ss.switchToSelectiveSearchQuality()
rects = ss.process()

(H, W) = image.shape[:2]
len_rects = len(rects)
search_time_end = time.time()

print("Predict...")
predict_time_start = time.time()

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
output = image.copy()

# plt.axis('off')
# plt.imshow(output)
# plt.show()

# ============================== Found Bounding Boxes ==========================================

inc_total = 0
inc_total_boxes = 0
inc_predict = 0
inc = 0

for (x, y, w, h) in rects:

    inc_total = inc_total + 1

    if object_w - offset_w < w < object_w + offset_w and object_h - offset_h < h < object_h + offset_h:
        boxes.append((x, y, w, h))
        inc_total_boxes = inc_total_boxes + 1

# ============================== Prepare Bounding Boxes ==========================================

# boxes = sort_boxes(boxes)

# for _, ii_box in enumerate(boxes):
#     print(ii_box[2] * ii_box[3])

boxes_m = merge_boxes(
    boxes=boxes,
    offset_x=box_offset_w,
    offset_y=box_offset_h,
)

# boxes_m = boxes

# ============================== Predict ========================================================

for (x, y, w, h) in boxes_m:

    roi = image[y:y + h, x:x + w]
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

    inc = inc + 1
    cv2.imwrite("garbage/" + str(inc) + "_g.jpg", roi)

    image_predict = canvas.paste_to_canvas(roi.copy())
    # print(image_predict.shape)
    img_array = tf.keras.utils.img_to_array(image_predict)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    # print(score)

    # inc = inc + 1
    # cv2.imwrite("garbage/" + str(inc) + "_g.jpg", image_predict)

    if class_names[np.argmax(score)] == "002" and (100 * np.max(score)) >= 50:
        inc_predict = inc_predict + 1
        print(score)
        # print(
        #     "This image most likely belongs to {} with a {:.2f} percent confidence."
        #     .format(class_names[np.argmax(score)], 100 * np.max(score))
        # )

        color = [random.randint(0, 255) for j in range(0, 3)]
        color = (0, 255, 0)
        cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)

        images_predicted.append(roi)
        score_predicted.append((100 * np.max(score)))

# =============================== Scores =====================================================

if score_predicted:
    max_score = np.max(score_predicted)
    print("Max score: ", max_score)

predict_time_end = time.time()
executionTime = (time.time() - startTime)
print("")
print('Search time in seconds: ' + str(search_time_end - startTime))
print('Predict time in seconds: ' + str(predict_time_end - predict_time_start))
print('Execution time in seconds: ' + str(executionTime))
print("")
print("Total: ", inc_total)
print("Total Boxes: ", inc_total_boxes)
print("Predict: ", inc_predict)

plt.axis('off')
plt.imshow(output)
plt.show()
