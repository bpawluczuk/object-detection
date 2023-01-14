import cv2
import numpy as np
from tensorflow import keras
import tensorflow as tf
from GradCAM import GradCAM
import matplotlib.pyplot as plt
# from vis.visualization import visualize_cam
# from vis.utils import utils

# ===============================================
model = keras.models.load_model('model/vgg')
print("Load model...")
# ===============================================

img_size = (256, 256)

last_conv_layer_name = "conv2d_8"

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

fig, ax = plt.subplots(1, 3)

ax[0].imshow(heatmap)
ax[1].imshow(image)
ax[2].imshow(output)
plt.show()

# Utility to search for layer index by name.
# Alternatively we can specify this as -1 since it corresponds to the last layer.
# layer_idx = utils.find_layer_idx(model, 'predictions')
# Swap softmax with linear
# model.layers[layer_idx].activation = keras.activations.linear
# model = utils.apply_modifications(model)
#
# penultimate_layer_idx = utils.find_layer_idx(model, "conv2d_8")
# class_idx = class_idxs_sorted[0]
# seed_input = image
# grad_top1 = visualize_cam(model, layer_idx, class_idx, seed_input,
#                           penultimate_layer_idx=penultimate_layer_idx,
#                           backprop_modifier=None,
#                           grad_modifier=None)
