import os
import cv2

img_height = 512
img_width = 512

image = cv2.imread("temp/003out/3_003.jpg")
roi = cv2.resize(image, (img_height, img_width))

cv2.imshow("Output", roi)
key = cv2.waitKey(0) & 0xFF
