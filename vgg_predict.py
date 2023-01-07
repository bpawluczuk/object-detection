import tensorflow as tf
import random
import cv2
import numpy as np
import seaborn as sns
import time

from tensorflow import keras

sns.set()
startTime = time.time()

# ===============================================================================================
img_height = 124
img_width = 124

model = keras.models.load_model('model/vgg')
print("Load model...")
# ===============================================================================================
images_predicted = []

class_names = ["000", "001", "002", "003", "004", "005"]
image = cv2.imread("images/shower_all.jpg")
# image = cv2.imread("images/003.jpg")
# image = cv2.imread("images/002.jpg")
# image = cv2.imread("images/003.jpg")
# image = cv2.imread("images/004.jpg")
# image = cv2.imread("images/005.jpg")

# Scale down to 25%
p = 0.35
w = int(image.shape[1] * p)
h = int(image.shape[0] * p)
image = cv2.resize(image, (w, h))

print("Search...")
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
ss.setBaseImage(image)
ss.switchToSelectiveSearchFast()
rects = ss.process()

(H, W) = image.shape[:2]
len_rects = len(rects)
search_time_end = time.time()

output = image.copy()
print("Predict...")
predict_time_start = time.time()

inc_total = 0
inc_pred = 0
inc = 0

for (x, y, w, h) in rects:

    inc_total = inc_total + 1

    # if w / float(W) < 0.4 or w / float(W) > 0.6 or h / float(H) < 0.7:
    #     continue

    # if w / float(W) < 0.03 or w / float(W) > 0.06 or h / float(H) < 0.05:
    #     continue

    roi = image[y:y + h, x:x + w]
    image_out = roi.copy()
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    roi = cv2.resize(roi, (img_height, img_width))

    img_array = tf.keras.utils.img_to_array(roi)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    print(score)

    if class_names[np.argmax(score)] == "003" and (100 * np.max(score)) >= 34:
        inc_pred = inc_pred + 1

        print(
            "This image most likely belongs to {} with a {:.2f} percent confidence."
            .format(class_names[np.argmax(score)], 100 * np.max(score))
        )

        color = [random.randint(0, 255) for j in range(0, 3)]
        cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)

        images_predicted.append(roi)

        inc = inc + 1
        cv2.imwrite("garbage/" + str(inc) + "_g.jpg", image_out)

predict_time_end = time.time()
executionTime = (time.time() - startTime)
print("")
print('Search time in seconds: ' + str(search_time_end - startTime))
print('Predict time in seconds: ' + str(predict_time_end - predict_time_start))
print('Execution time in seconds: ' + str(executionTime))
print("")
print("Total: ", inc_total)
print("Predict: ", inc_pred)

cv2.imshow("Output", output)
key = cv2.waitKey(0) & 0xFF
