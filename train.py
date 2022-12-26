import tensorflow as tf
import datetime
import cv2
import numpy as np
import seaborn as sns

from tensorflow import keras
from keras.layers import *
from keras.optimizers import Adam, SGD
from keras.models import Model
import matplotlib.pyplot as plt

sns.set()

# =========================================================
data_dir = "dataset"

batch_size = 1
num_classes = 6
epochs = 15

img_height = 224
img_width = 224
channels = 3

IMAGE_SHAPE = (img_height, img_width, channels)
# =========================================================

train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    shuffle=True,
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

validation_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    shuffle=True,
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

class_names = train_ds.class_names
print(class_names)

for image_batch, labels_batch in train_ds:
    print(image_batch.shape)
    print(labels_batch.shape)
    break

# =========================================================

# optimizer = Adam(learning_rate=2e-5, beta_1=0.5)
optimizer = SGD(learning_rate=1e-6, momentum=0.9)

kernel_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

# =========================================================

input = Input(shape=IMAGE_SHAPE)

x = RandomFlip("horizontal", input_shape=IMAGE_SHAPE)(input)
x = RandomRotation(0.1)(x)
x = RandomZoom(0.1)(x)

x = Rescaling(1. / 255)(x)

x = Conv2D(64, kernel_size=3, padding='same', activation='relu')(x)
x = Conv2D(64, kernel_size=3, padding='same', activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

x = Conv2D(128, kernel_size=3, padding='same', activation='relu')(x)
x = Conv2D(128, kernel_size=3, padding='same', activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

x = Conv2D(256, kernel_size=3, padding='same', activation='relu')(x)
x = Conv2D(256, kernel_size=3, padding='same', activation='relu')(x)
x = Conv2D(256, kernel_size=3, padding='same', activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

x = Conv2D(512, kernel_size=3, padding='same', activation='relu')(x)
x = Conv2D(512, kernel_size=3, padding='same', activation='relu')(x)
x = Conv2D(512, kernel_size=3, padding='same', activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

x = Conv2D(512, kernel_size=3, padding='same', activation='relu')(x)
x = Conv2D(512, kernel_size=3, padding='same', activation='relu')(x)
x = Conv2D(512, kernel_size=3, padding='same', activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

# x = Dropout(0.2)(x)
x = Flatten()(x)
x = Dense(256, activation='relu')(x)
x = Dense(128, activation='relu')(x)
output = Dense(num_classes, "softmax", name="predictions")(x)

model = Model(inputs=input, outputs=output, name='Custom_model')

# summarize layers
print(model.summary())

model.compile(optimizer=optimizer, loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])

# Train the Model
history = model.fit(train_ds,
                    epochs=epochs,
                    validation_data=validation_ds)

model.save("model")
print("save model")

# =========================================================
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
# =========================================================

img_path = "dataset/004/16_imagea.jpg"
# img_path = "dataset/train/005/55_image.jpg"

img = tf.keras.utils.load_img(
    img_path, target_size=(img_height, img_width)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)
