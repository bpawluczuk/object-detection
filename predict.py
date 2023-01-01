import tensorflow as tf
import random
import cv2
import numpy as np
import seaborn as sns

from tensorflow import keras

sns.set()

# ===============================================================================================
img_height = 124
img_width = 124

model = keras.models.load_model('model')
print("load model")
# ===============================================================================================

# class_names = ["001", "006", "003", "004", "005", "006"]
class_names = ["000", "001", "006"]

image = cv2.imread("images/shower_all.jpg")
image = cv2.resize(image, (600, 800), interpolation=cv2.INTER_AREA)

# cv2.imshow("Output", image)
# key = cv2.waitKey(0) & 0xFF

ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
ss.setBaseImage(image)
ss.switchToSelectiveSearchFast()
rects = ss.process()

(H, W) = image.shape[:2]
len_rects = len(rects)

output = image.copy()
print("Predict...")

inc_total = 0
inc_pred = 0

for (x, y, w, h) in rects:

    inc_total = inc_total + 1

    if w / float(W) < 0.05 or w / float(W) > 0.1 or h / float(H) < 0.08:
        continue

    inc_pred = inc_pred + 1

    roi = image[y:y + h, x:x + w]
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    roi = cv2.resize(roi, (img_height, img_width))

    img_array = tf.keras.utils.img_to_array(roi)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    print(score)
    if class_names[np.argmax(score)] == "006" and (100 * np.max(score)) >= 40:
        print(
            "This image most likely belongs to {} with a {:.2f} percent confidence."
            .format(class_names[np.argmax(score)], 100 * np.max(score))
        )

        color = [random.randint(0, 255) for j in range(0, 3)]
        cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)

print("Total: ", inc_total)
print("Predict: ", inc_pred)

cv2.imshow("Output", output)
key = cv2.waitKey(0) & 0xFF
