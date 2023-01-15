import tensorflow as tf
import random
import cv2
import numpy as np
import seaborn as sns
import time
import matplotlib.pyplot as plt
from tensorflow import keras

sns.set()
startTime = time.time()

# ===============================================================================================
img_height = 512
img_width = 512
channels = 3

canvas = np.zeros((img_width, img_height, channels), np.uint8)

# ===============================================================================================
image = cv2.imread("garbage/9_g.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
(H, W) = image.shape[:2]

plt.imshow(image)
plt.show()

ph = img_height / float(H)
w = int(image.shape[1] * ph)
h = int(image.shape[0] * ph)
new_image = cv2.resize(image, (w, h))

plt.imshow(new_image)
plt.show()

print(new_image.shape[0], canvas.shape[0])
print(new_image.shape[1], canvas.shape[1])

y_off = round((canvas.shape[0] - new_image.shape[0]) / 2)
x_off = round((canvas.shape[1] - new_image.shape[1]) / 2)
print(y_off, x_off)

result = canvas.copy()
result[y_off:y_off + new_image.shape[0], x_off:x_off + new_image.shape[1]] = new_image

plt.imshow(result)
plt.show()
