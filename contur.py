import cv2
import matplotlib.pyplot as plt
import numpy as np

from Canvas import Canvas


def get_contour_areas(contours):
    all_areas = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        all_areas.append(area)
    return all_areas


# The local path to our target image
img_path = "images/6.jpg"
image_out = "dataset/001/61.jpg"

# load the input image
source_image = cv2.imread(img_path)

blurred = cv2.GaussianBlur(source_image, (5, 5), 0)

# convert the input image to grayscale
gray = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY)

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

mask = np.zeros(source_image.shape[:2], np.uint8)
mask.fill(1)

mask = cv2.drawContours(mask, largest_item, -1, (0, 0, 0), 1)
cv2.fillConvexPoly(mask, largest_item, 255)

plt.imshow(mask)
plt.show()

mask_inv = cv2.bitwise_not(mask)

plt.imshow(mask_inv)
plt.show()

result = cv2.imread(img_path)

plt.imshow(result)
plt.show()

masked = cv2.bitwise_and(result, result, mask=mask)

plt.imshow(masked)
plt.show()

# ==========================================

x, y, w, h = cv2.boundingRect(largest_item)
color = (0, 255, 0)
cv2.rectangle(masked, (x, y), (x + w, y + h), color, 2)

plt.imshow(masked)
plt.show()

cropped_image = masked[y:y + h, x:x + w]

plt.imshow(cropped_image)
plt.show()

canvas = Canvas((512, 512, 3))
result = canvas.paste_to_canvas(cropped_image)

plt.imshow(result)
plt.show()

cv2.imwrite(image_out, result)
