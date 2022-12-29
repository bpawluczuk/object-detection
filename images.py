import os
import cv2

source_dir = "shauma/006/"
dest_dir = "dataset_test/006/"

inc = 189
for file in os.listdir(source_dir):
    inc = inc + 1

    source_image = cv2.imread(source_dir + file)

    if source_image is not None and source_image.any():
        cv2.imwrite(dest_dir + str(inc) + "_006.jpg", source_image)
