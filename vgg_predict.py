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

class_names = ["001", "002"]
image = cv2.imread("images/test_shape.jpg")

# Scale down
# p = 0.50
# w = int(image.shape[1] * p)
# h = int(image.shape[0] * p)
# image = cv2.resize(image, (w, h))

print("Search...")

ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
ss.setBaseImage(image)
ss.switchToSelectiveSearchFast()
# ss.switchToSelectiveSearchQuality()
rects = ss.process()

(H, W) = image.shape[:2]
len_rects = len(rects)
search_time_end = time.time()

print("Predict...")
predict_time_start = time.time()

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
output = image.copy()
plt.imshow(output)
plt.show()

inc_total = 0
inc_pred = 0
inc = 0

for (x, y, w, h) in rects:

    inc_total = inc_total + 1

    if w > 200 or w < 140 or h > 450 or h < 300:
        continue

    roi = image[y:y + h, x:x + w]
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

    image_predict = canvas.paste_to_canvas(roi.copy())

    img_array = tf.keras.utils.img_to_array(image_predict)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    print(score)

    # inc = inc + 1
    # cv2.imwrite("garbage/" + str(inc) + "_g.jpg", image_predict)

    if class_names[np.argmax(score)] == "001" and (100 * np.max(score)) >= 45:
        inc_pred = inc_pred + 1

        print(
            "This image most likely belongs to {} with a {:.2f} percent confidence."
            .format(class_names[np.argmax(score)], 100 * np.max(score))
        )

        color = [random.randint(0, 255) for j in range(0, 3)]
        cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)

        images_predicted.append(roi)
        score_predicted.append((100 * np.max(score)))

        # inc = inc + 1
        # cv2.imwrite("garbage/" + str(inc) + "_g.jpg", image_out)

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
print("Predict: ", inc_pred)

plt.imshow(output)
plt.show()
