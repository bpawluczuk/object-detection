import cv2
import numpy as np

# wczytanie obrazu
image = cv2.imread('images/shelf_3.jpg')

(H, W) = image.shape[:2]
print(H, W)

# keep a copy of the original image
orig_image = image.copy()
# convert to RGB image and convert to float32
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = image.astype(np.float32) / 255.0
# grayscale and blurring for canny edge detection
gray = cv2.cvtColor(orig_image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
canny = cv2.Canny(blurred, 50, 200)

# ustawienia dla createStructuredEdgeDetection
model = cv2.ximgproc.createStructuredEdgeDetection('model/model.yml')

# wygenerowanie mapy orientacji krawędzi
edges = model.detectEdges(image)

orimap = model.computeOrientation(edges)
edges = model.edgesNms(edges, orimap)

edge_boxes = cv2.ximgproc.createEdgeBoxes()
# edge_boxes.setMaxBoxes(1)
boxes, _ = edge_boxes.getBoundingBoxes(edges, orimap)

for box in boxes:
    x, y, w, h = box

    if (w / float(W) < 0.06 or w / float(W) > 0.08) or (h / float(H) < 0.6 or h / float(H) > 0.9):
        continue

    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1, cv2.LINE_AA)

cv2.imshow('Result', image)
cv2.imshow('Structured forests', edges)
cv2.imshow('Canny', canny)
cv2.waitKey(0)
