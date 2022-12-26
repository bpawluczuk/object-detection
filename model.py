import tensorflow as tf
import datetime
import cv2
import numpy as np
import seaborn as sns

from tensorflow import keras
from keras.layers import *
from keras.optimizers import Adam
from keras.models import Model
from keras.utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image

sns.set()

# =========================================================
batch_size = 1

size = 224
width = 224
channels = 3

IMAGE_SHAPE = (size, size, channels)
# =========================================================

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    'dataset/train',
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    'dataset/validation',
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical')

# =========================================================

optimizer = Adam(learning_rate=2e-5, beta_1=0.5)
kernel_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

# =========================================================

input = Input(shape=IMAGE_SHAPE)

x = Conv2D(32, kernel_size=4, activation='relu')(input)
x = MaxPooling2D(pool_size=(2, 2))(x)

x = Conv2D(32, kernel_size=4, activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

x = Conv2D(16, kernel_size=4, activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

x = Flatten()(x)
x = Dense(100, activation='relu')(x)
output = Dense(5, "softmax", name="predictions")(x)

model = Model(inputs=input, outputs=output, name='Custom_model')

# summarize layers
print(model.summary())

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=["accuracy"])

# Train the Model
history = model.fit(train_generator,
                    epochs=50,
                    validation_data=validation_generator)

# =========================================================

# image = cv2.imread("dataset/train/002.shauma/30_image.jpg")
image = cv2.imread("dataset/train/005.shauma/55_image.jpg")

test_input = cv2.resize(image, (224, 224))

img_array = np.asarray(test_input)
img_batch = np.expand_dims(img_array / 255, axis=0)
print(img_batch.shape)

predict = model.predict(img_batch)
print(predict)
