import random
import cv2

image = cv2.imread("dataset/001/20_image.jpg")

output = cv2.resize(image, (224, 224))

cv2.imshow("Input", image)
cv2.imshow("Output", output)
key = cv2.waitKey(0) & 0xFF
