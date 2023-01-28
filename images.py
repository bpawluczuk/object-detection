import os
import cv2

source_dir = "dataset/002/"
dest_dir = "dataset/002/"

inc = 0
for file in os.listdir(source_dir):

    inc = inc + 1

    source_image = cv2.imread(source_dir + file)

    if source_image is not None and source_image.any():
        # cv2.convertScaleAbs(source_image, source_image, 1, -20)
        cv2.imwrite(dest_dir + str(inc) + ".jpg", source_image)
