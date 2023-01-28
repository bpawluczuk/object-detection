import cv2
import numpy as np


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
img_path = "output/s222.jpg"
image_out = "output/555.jpg"

# load the input image
source_image = cv2.imread(img_path)
# source_image = change_brightness(source_image, -70)
cv2.imshow("source_image", source_image)
cv2.waitKey(0)

blurred = cv2.GaussianBlur(source_image, (3, 3), 0)
cv2.imshow("blurred", blurred)
cv2.waitKey(0)

edged = cv2.Canny(blurred, 10, 100)
cv2.imshow("edged", edged)
cv2.waitKey(0)

# convert the input image to grayscale
# gray = cv2.cvtColor(edged, cv2.COLOR_BGR2GRAY)

# apply thresholding to convert grayscale to binary image
# ret, thresh = cv2.threshold(edged, 150, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
dilate = cv2.dilate(edged, kernel, iterations=1)
cv2.imshow("dilate", dilate)
cv2.waitKey(0)

# find the contours
contours, hierarchy = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print("Number of contours detected:", len(contours))

# Find the convex hull for all the contours
for cnt in contours:
    hull = cv2.convexHull(cnt)
    img = cv2.drawContours(source_image, [cnt], 0, (0, 255, 0), 2)
    # img = cv2.drawContours(img, [hull], 0, (255, 0, 0), 3)

sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
largest_item = sorted_contours[0]

img = cv2.drawContours(source_image, largest_item, -1, (0, 0, 255), 10)

cv2.imshow("mask_inv", img)
cv2.waitKey(0)

mask = np.zeros(source_image.shape[:2], np.uint8)
mask.fill(1)

mask = cv2.drawContours(mask, largest_item, -1, (0, 0, 0), 1)
cv2.fillConvexPoly(mask, largest_item, 255)

mask_inv = cv2.bitwise_not(mask)
cv2.imshow("mask_inv", mask)
cv2.waitKey(0)

result = cv2.imread(img_path)
cv2.bitwise_or(result, result, mask=mask_inv)
result[mask_inv != 0] = 0

cv2.imshow("masked", result)
cv2.waitKey(0)
