import cv2
import numpy as np
from tensorflow import keras
import tensorflow as tf
from GradCAM import GradCAM
import matplotlib.pyplot as plt

# ===============================================
model = keras.models.load_model('model/vgg')
print("Load model...")
# ===============================================

img_size = (512, 512)

last_conv_layer_name = "conv2d_17"

# The local path to our target image
img_path = "images/001.jpg"

image = cv2.imread(img_path)
image = cv2.resize(image, img_size)
img_array = tf.keras.utils.img_to_array(image)
img_array = tf.expand_dims(img_array, 0)

preds = model.predict(img_array)
i = np.argmax(preds[0])

for idx in range(len(model.layers)):
    print(model.get_layer(index=idx).name)

icam = GradCAM(model, i, last_conv_layer_name)
heatmap = icam.compute_heatmap(img_array)
heatmap = cv2.resize(heatmap, (256, 256))

image = cv2.imread(img_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (256, 256))
print(heatmap.shape, image.shape)

(heatmap, output) = icam.overlay_heatmap(heatmap, image, alpha=0.5)

fig = plt.figure(figsize=(20, 20))
fig, ax = plt.subplots(1, 3)

ax[0].imshow(heatmap)
ax[1].imshow(image)
ax[2].imshow(output)
plt.show()
