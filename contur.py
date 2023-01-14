import cv2
import matplotlib.pyplot as plt
import numpy as np


def get_contour_areas(contours):
    all_areas = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        all_areas.append(area)
    return all_areas


# The local path to our target image
img_path = "images/00093.jpg"

# load the input image
source_image = cv2.imread(img_path)
blurred = cv2.GaussianBlur(source_image, (5, 5), 0)

# convert the input image to grayscale
gray = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY)

gray[gray < 5] = 255.0

# apply thresholding to convert grayscale to binary image
ret, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# find the contours
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
print("Number of contours detected:", len(contours))

# display the image with drawn contour and convex hull

plt.imshow(gray, cmap="gray")
plt.show()

# Find the convex hull for all the contours
for cnt in contours:
    hull = cv2.convexHull(cnt)
    img = cv2.drawContours(source_image, [cnt], 0, (0, 255, 0), 2)
    # img = cv2.drawContours(img, [hull], 0, (255, 0, 0), 3)

plt.imshow(img, cmap="gray")
plt.show()

sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
largest_item = sorted_contours[0]

img = cv2.drawContours(source_image, largest_item, -1, (0, 0, 255), 10)
plt.imshow(img, cmap="gray")
plt.show()

mask = np.zeros(source_image.shape, np.uint8)
mask.fill(255)
mask = cv2.drawContours(mask, largest_item, -1, (0, 0, 0), 1)

cv2.fillConvexPoly(mask, largest_item, 0)

plt.imshow(mask, cmap="gray")
plt.show()

# ==========================================
