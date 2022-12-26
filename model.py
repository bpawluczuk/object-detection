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
import matplotlib.pyplot as plt

sns.set()

# =========================================================
data_dir = "flowers"
train_data_dir = "dataset/train"
validation_data_dir = "dataset/validation"

batch_size = 1
num_classes = 5
epochs = 10

img_height = 224
img_width = 224
channels = 3

IMAGE_SHAPE = (img_height, img_width, channels)
# =========================================================

train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

validation_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
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

optimizer = Adam(learning_rate=2e-5, beta_1=0.5)
kernel_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

# =========================================================

input = Input(shape=IMAGE_SHAPE)

x = RandomFlip("horizontal", input_shape=IMAGE_SHAPE)(input)
x = RandomRotation(0.1)(x)
x = RandomZoom(0.1)(x)

x = Rescaling(1. / 255)(x)
x = Conv2D(16, kernel_size=4, padding='same', activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

x = Conv2D(32, kernel_size=4, padding='same', activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

x = Conv2D(64, kernel_size=4, padding='same', activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

x = Dropout(0.2)(x)
x = Flatten()(x)
x = Dense(244, activation='relu')(x)
output = Dense(num_classes, "softmax", name="predictions")(x)

model = Model(inputs=input, outputs=output, name='Custom_model')

# summarize layers
print(model.summary())

model.compile(optimizer=optimizer, loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])

# Train the Model
history = model.fit(train_ds,
                    epochs=epochs,
                    validation_data=validation_ds)

model.save_weights("model/flowers.h5")
print("save model weights")

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

img_path = "flowers/dandelion/15987457_49dc11bf4b.jpg"
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
