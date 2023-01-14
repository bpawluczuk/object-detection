import cv2
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow import keras
from keras.optimizers import Adam
import numpy as np

from GradCAM import GradCAM

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# train set / data
x_train = x_train.astype('float32') / 255
# train set / target
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)

# validation set / data
x_test = x_test.astype('float32') / 255
# validation set / target
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

input = tf.keras.Input(shape=(32, 32, 3))
efnet = tf.keras.applications.EfficientNetB0(weights='imagenet',
                                             include_top=False,
                                             input_tensor=input)
# Now that we apply global max pooling.
gap = tf.keras.layers.GlobalMaxPooling2D()(efnet.output)

# Finally, we add a classification layer.
output = tf.keras.layers.Dense(10, activation='softmax')(gap)

# bind all
func_model = tf.keras.Model(efnet.input, output)

func_model.compile(
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=tf.keras.metrics.CategoricalAccuracy(),
    optimizer=Adam())

# fit
func_model.fit(x_train, y_train, batch_size=128, epochs=15, verbose=2)
# tf.keras.models.save_model(func_model, "model/example")
# print("Save model")

# func_model = tf.saved_model.load('model/example')
# print("Load model...")

image = cv2.imread('images/dog.jpg')
image = cv2.resize(image, (32, 32))
image = image.astype('float32') / 255
image = np.expand_dims(image, axis=0)

preds = func_model.predict(image)
i = np.argmax(preds[0])

for idx in range(len(func_model.layers)):
    print(func_model.get_layer(index=idx).name)

icam = GradCAM(func_model, i, 'block5c_project_conv')
heatmap = icam.compute_heatmap(image)
heatmap = cv2.resize(heatmap, (32, 32))

image = cv2.imread('images/dog.jpg')
image = cv2.resize(image, (32, 32))
print(heatmap.shape, image.shape)

(heatmap, output) = icam.overlay_heatmap(heatmap, image, alpha=0.5)

fig, ax = plt.subplots(1, 3)

ax[0].imshow(heatmap)
ax[1].imshow(image)
ax[2].imshow(output)
plt.show()
