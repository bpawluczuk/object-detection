import os
import cv2

img_height = 512
img_width = 512


def change_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.add(v, value)
    v[v > 255] = 255
    v[v < 0] = 0
    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


def fast_brightness(input_image, brightness=30):
    """ input_image:  color or grayscale image
        brightness:  -255 (all black) to +255 (all white)

        returns image of same type as input_image but with
        brightness adjusted"""
    img = input_image.copy()
    cv2.convertScaleAbs(img, img, 1, brightness)
    return img


image = cv2.imread("dataset/001/21.jpg")
cv2.imshow("Input", image)

image1 = change_brightness(image.copy(), -50)
cv2.imshow("Output 1", image1)

image2 = fast_brightness(image.copy(), -50)
cv2.imshow("Output fast", image2)

key = cv2.waitKey(0) & 0xFF
