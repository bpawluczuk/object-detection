import tensorflow as tf
import seaborn as sns

from tensorflow import keras
from keras.layers import *
from keras.optimizers import Adam, SGD
from keras.models import Model
import matplotlib.pyplot as plt

sns.set()

# =========================================================
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

devices = session.list_devices()
for d in devices:
    print(d.name)

physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    pass

# =========================================================

data_dir = "dataset"
data_valid_dir = "dataset_valid"

batch_size = 8
num_classes = 2
epochs = 100

img_height = 512
img_width = 512
channels = 3

IMAGE_SHAPE = (img_height, img_width, channels)
# =========================================================

train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.3,
    subset="training",
    shuffle=True,
    seed=1,
    image_size=(img_height, img_width),
    batch_size=batch_size)

validation_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.3,
    subset="validation",
    shuffle=True,
    seed=1,
    image_size=(img_height, img_width),
    batch_size=batch_size)

class_names = train_ds.class_names
print(class_names)

for image_batch, labels_batch in train_ds:
    print(image_batch.shape)
    print(labels_batch.shape)
    break


# =========================================================

# for step, (x_batch_train, y_batch_train) in enumerate(train_ds):
#     plt.figure(figsize=(10, 10))
#     for image_batch, label_batch in train_ds.take(step):
#         for i in range(2):
#             ax = plt.subplot(2, 2, i + 1)
#             plt.imshow(image_batch[i].numpy().astype("uint8"))
#             plt.title(class_names[label_batch[i]])
#             plt.axis("off")
#         plt.show()

# =========================================================

# optimizer = Adam(learning_rate=1e-6, beta_1=0.5)
optimizer = SGD(learning_rate=1e-6, momentum=0.9)
# optimizer = SGD(learning_rate=2e-5, momentum=0.9)

kernel_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

# =========================================================

input = Input(shape=IMAGE_SHAPE)

# x = RandomFlip("horizontal", input_shape=IMAGE_SHAPE)(input)
# x = RandomRotation(0.2)(x)
# x = RandomZoom(0.2)(x)

x = Rescaling(1. / 255)(input)

x = Conv2D(64, kernel_size=3, padding='same', activation='relu')(x)
x = Conv2D(64, kernel_size=3, padding='same', activation='relu')(x)
x = Conv2D(64, kernel_size=3, padding='same', activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

x = Conv2D(128, kernel_size=3, padding='same', activation='relu')(x)
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

x = BatchNormalization()(x)

x = Dropout(0.1)(x)
x = Flatten()(x)

x = Dense(128, activation='relu')(x)
x = Dense(64, activation='relu')(x)
x = Dense(32, activation='relu')(x)

output = Dense(num_classes, "softmax", name="predictions")(x)

model = Model(inputs=input, outputs=output, name='Custom_model')

# summarize layers
print(model.summary())

model.compile(optimizer=optimizer, loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])

# Train the Model
history = model.fit(train_ds,
                    epochs=epochs,
                    validation_data=validation_ds)

model.save("model/vgg")
print("Save model")

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

# img_path = "_dataset/001/1_001.jpg"
#
# img = tf.keras.utils.load_img(
#     img_path, target_size=(img_height, img_width)
# )
# img_array = tf.keras.utils.img_to_array(img)
# img_array = tf.expand_dims(img_array, 0)  # Create a batch
#
# predictions = model.predict(img_array)
# score = tf.nn.softmax(predictions[0])
#
# print(
#     "This image most likely belongs to {} with a {:.2f} percent confidence."
#     .format(class_names[np.argmax(score)], 100 * np.max(score))
# )
#
# print(score)
