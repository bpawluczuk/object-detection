import tensorflow as tf
import random
import cv2
import numpy as np
import seaborn as sns

from tensorflow import keras
from keras.layers import *
from keras.optimizers import Adam
from keras.models import Model

sns.set()

# =========================================================
batch_size = 1
num_classes = 4

img_height = 224
img_width = 224
channels = 3

IMAGE_SHAPE = (img_height, img_width, channels)
# =========================================================

optimizer = Adam(learning_rate=2e-5, beta_1=0.5)
kernel_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

# =========================================================

input = Input(shape=IMAGE_SHAPE)

x = RandomFlip("horizontal", input_shape=IMAGE_SHAPE)(input)
x = RandomRotation(0.1)(x)
x = RandomZoom(0.1)(x)

x = Rescaling(1. / 255)(x)
x = Conv2D(16, kernel_size=3, padding='same', activation='relu')(x)
x = MaxPooling2D()(x)

x = Conv2D(32, kernel_size=3, padding='same', activation='relu')(x)
x = MaxPooling2D()(x)

x = Conv2D(64, kernel_size=3, padding='same', activation='relu')(x)
x = MaxPooling2D()(x)

x = Dropout(0.2)(x)
x = Flatten()(x)
x = Dense(122, activation='relu')(x)
output = Dense(num_classes, "softmax", name="predictions")(x)

model = Model(inputs=input, outputs=output, name='Custom_model')

# summarize layers
print(model.summary())

model.compile(optimizer=optimizer, loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])

model.load_weights("model/flowers.h5")
print("load model weights")

# ===============================================================================================

class_names = ["001", "002", "003", "004"]

image = cv2.imread("images/test_półka.jpg")
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
ss.setBaseImage(image)
ss.switchToSelectiveSearchFast()
rects = ss.process()

(H, W) = image.shape[:2]
len_rects = len(rects)

output = image.copy()

for (x, y, w, h) in rects:

    if w / float(W) < 0.1 or h / float(H) < 0.1:
        continue

    roi = image[y:y + h, x:x + w]
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    roi = cv2.resize(roi, (224, 224))

    img_array = tf.keras.utils.img_to_array(roi)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    if class_names[np.argmax(score)] == "003" and (100 * np.max(score)) > 47.4:
        print(
            "This image most likely belongs to {} with a {:.2f} percent confidence."
            .format(class_names[np.argmax(score)], 100 * np.max(score))
        )

        color = [random.randint(0, 255) for j in range(0, 3)]
        cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)

cv2.imshow("Output", output)
key = cv2.waitKey(0) & 0xFF
