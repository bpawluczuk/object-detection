import os
import cv2

source_dir = "dataset/000-kopia 5/"
dest_dir = "dataset/000/"

inc = 1082
for file in os.listdir(source_dir):
    inc = inc + 1

    source_image = cv2.imread(source_dir + file)

    if source_image is not None and source_image.any():
        cv2.imwrite(dest_dir + str(inc) + "_000.jpg", source_image)
