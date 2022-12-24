import random
import cv2

image = cv2.imread("images/bottle.jpeg")

ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
ss.setBaseImage(image)
ss.switchToSelectiveSearchQuality()
rects = ss.process()

for i in range(0, len(rects), 100):
    # clone the original image so we can draw on it
    output = image.copy()
    # loop over the current subset of region proposals
    for (x, y, w, h) in rects[i:i + 100]:
        if (w > 20 and w < 30):
            continue
        # draw the region proposal bounding box on the image
        color = [random.randint(0, 255) for j in range(0, 3)]
        cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)
    # show the output image
        cv2.imshow("Output", output)
        key = cv2.waitKey(0) & 0xFF
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
