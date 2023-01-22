import random
import cv2

image = cv2.imread("images/6.jpg")

# image = image.copy()
# image = cv2.resize(image, (800, 1200), interpolation=cv2.INTER_AREA)
# cv2.imwrite("output/" + "resized_image.jpg", image)

ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
ss.setBaseImage(image)
# ss.switchToSelectiveSearchQuality()
ss.switchToSelectiveSearchFast()
rects = ss.process()

(H, W) = image.shape[:2]
len_rects = len(rects)

output = image.copy()

inc = 0
for (x, y, w, h) in rects:

    # if (w / float(W) < 0.06 or w / float(W) > 0.08) or (h / float(H) < 0.1 or h / float(H) > 0.2):
    #     continue

    # if not (w / float(W) >= 0.2 and w / float(W) <= 0.6 and h / float(H) >= 0.2 and h / float(H) <= 1):
    #     continue

    # if not (w / float(W) >= 0.06 and w / float(W) <= 0.08 and h / float(H) >= 0.8 and h / float(H) <= 1):
    #     continue

    # if not (w / float(W) > 0.02 and w / float(W) < 0.07 and h / float(H) > 0.06 and h / float(H) < 0.15):
    #     continue

    # if not (w / float(W) < 0.1 and h / float(H) > 0.8):
    #     continue

    inc = inc + 1
    # draw the region proposal bounding box on the image
    color = [random.randint(0, 255) for j in range(0, 3)]
    cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)

    roi = image[y:y + h, x:x + w]
    cv2.imwrite("output/" + str(inc) + "_000.jpg", roi)

print(inc)
cv2.imshow("Output", output)
key = cv2.waitKey(0) & 0xFF
