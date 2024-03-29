import cv2
import numpy as np
from scipy.interpolate import splprep, splev

from Canvas import Canvas


def get_contour_areas(contours):
    all_areas = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        all_areas.append(area)
    return all_areas


def change_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.add(v, value)
    v[v > 255] = 255
    v[v < 0] = 0
    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


# The local path to our target image
img_path = "images/444.jpg"
image_out = "output/444.jpg"

# load the input image
source_image = cv2.imread(img_path)
# source_image = change_brightness(source_image, 70)
cv2.imshow("source_image", source_image)
cv2.waitKey(0)

# filter = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
# # Applying cv2.filter2D function on our Cybertruck image
# source_image = cv2.filter2D(source_image, -1, filter)
# cv2.imshow("sharpen_img_1", source_image)
# cv2.waitKey(0)

blurred = cv2.GaussianBlur(source_image, (3, 3), 0)
# blurred = cv2.medianBlur(source_image, 5)
# blurred = cv2.addWeighted( blurred, 1.5, blurred, -0.5, 0)
cv2.imshow("blurred", blurred)
cv2.waitKey(0)

edged = cv2.Canny(blurred, 10, 60)
cv2.imshow("edged", edged)
cv2.waitKey(0)

# convert the input image to grayscale
# gray = cv2.cvtColor(edged, cv2.COLOR_BGR2GRAY)

# apply thresholding to convert grayscale to binary image
# ret, thresh = cv2.threshold(edged, 150, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
dilate = cv2.dilate(edged, kernel, iterations=1)
cv2.imshow("dilate", dilate)
cv2.waitKey(0)

# find the contours
contours, hierarchy = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print("Number of contours detected:", len(contours))

# Find the convex hull for all the contours
for cnt in contours:
    hull = cv2.convexHull(cnt)
    # img = cv2.drawContours(source_image, [cnt], 0, (0, 255, 0), 10)
    # img = cv2.drawContours(source_image, [hull], 0, (255, 0, 0), 10)

sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
largest_item = sorted_contours[0]

# epsilon = 0.01 * cv2.arcLength(largest_item, True)
# largest_item = cv2.approxPolyDP(largest_item, epsilon, True)

img = cv2.drawContours(source_image, largest_item, -1, (0, 0, 255), 10)
cv2.imshow("mask_inv", img)
cv2.waitKey(0)

mask = np.zeros(source_image.shape[:2], np.uint8)
mask.fill(1)

mask = cv2.drawContours(mask, largest_item, -1, (0, 0, 0), 10)
cv2.fillConvexPoly(mask, largest_item, 255)

mask_inv = cv2.bitwise_not(mask)
cv2.imshow("mask_dilate", mask)
cv2.waitKey(0)

result = cv2.imread(img_path)
cv2.bitwise_or(result, result, mask=mask_inv)
result[mask_inv != 0] = 0

cv2.imshow("masked", result)
cv2.waitKey(0)

# ==========================================

x, y, w, h = cv2.boundingRect(largest_item)
color = (0, 255, 0)
cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)

cv2.imshow("rect", result)
cv2.waitKey(0)

cropped_image = result[y:y + h, x:x + w]

cv2.imshow("crop", result)
cv2.waitKey(0)

canvas = Canvas((512, 512, 3))
result = canvas.paste_to_canvas(cropped_image)

cv2.imshow("canvas", result)
cv2.waitKey(0)

cv2.imwrite(image_out, result)
