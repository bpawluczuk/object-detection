import cv2
import os

name = "002"
source_dir = "temp/" + name + "test/"
dest_dir = "temp/" + name + "out/"

# source_dir = "temp/test/"

inc = 0
n = 0
for file in os.listdir(source_dir):

    if file.startswith('.'):
        continue

    print(">>> ", source_dir + file)

    image = cv2.imread(source_dir + file)
    # Scale down
    p = 0.15
    w = int(image.shape[1] * p)
    h = int(image.shape[0] * p)
    image = cv2.resize(image, (w, h))
    # ===========

    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)
    ss.switchToSelectiveSearchQuality()
    # ss.switchToSelectiveSearchFast()
    rects = ss.process()

    (H, W) = image.shape[:2]
    len_rects = len(rects)

    output = image.copy()

    for (x, y, w, h) in rects:

        # n = n + 1
        # if not (n % 4 == 0):
        #     continue

        # if (w / float(W) < 0.6 or w / float(W) > 0.7) or (h / float(H) < 0.7):
        #     continue

        if w / float(W) < 0.5 or h / float(H) < 0.4:
            continue

        inc = inc + 1
        roi = image[y:y + h, x:x + w]
        dest_path = dest_dir + str(inc) + "_" + name + "2.jpg"
        cv2.imwrite(dest_path, roi)
        print(dest_path)
