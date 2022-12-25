import tensorflow as tf
import datetime
import cv2
import numpy
import seaborn as sns

from tensorflow import keras
from keras.layers import *
from keras.optimizers import Adam
from keras.models import Model
from keras.utils import plot_model
from keras.preprocessing.image import ImageDataGenerator

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
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    'dataset/validation',
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='binary')

# =========================================================

optimizer = Adam(learning_rate=2e-5, beta_1=0.5)
kernel_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

# =========================================================

input = Input(shape=IMAGE_SHAPE)

conv1 = Conv2D(32, kernel_size=4, activation='relu')(input)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

conv2 = Conv2D(16, kernel_size=4, activation='relu')(pool1)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

flat = Flatten()(pool2)
hidden1 = Dense(10, activation='relu')(flat)
output = Dense(3, "softmax", name="predictions")(hidden1)

model = Model(inputs=input, outputs=output, name='Custom_model')

# summarize layers
print(model.summary())

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=["accuracy"])

# Train the Model
history = model.fit(train_generator,
                    epochs=30,
                    validation_data=validation_generator)
