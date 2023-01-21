import cv2
from matplotlib import pyplot as plt

from Canvas import Canvas

img_height = 512
img_width = 512
channels = 3
CANVAS_SHAPE = (img_height, img_width, channels)

canvas = Canvas(CANVAS_SHAPE)

image_path = "images/shape_2_1.jpg"
image = cv2.imread(image_path)

plt.imshow(image)
plt.show()

result = canvas.paste_to_canvas(image)
cv2.imwrite("dataset/002/1.jpg", result)

plt.imshow(image)
plt.show()