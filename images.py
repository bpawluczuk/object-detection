import os
import cv2

source_dir = "temp/"
dest_dir = "temp/003/"

inc = 0
for file in os.listdir(source_dir):

    for n in range(100):
        inc = inc + 1

        source_image = cv2.imread(source_dir + file)

        if source_image is not None and source_image.any():
            cv2.imwrite(dest_dir + str(inc) + "_003a.jpg", source_image)
