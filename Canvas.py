import cv2
import numpy as np
import matplotlib.pyplot as plt


class Canvas:
    def __init__(self, canvas_shape, image_path):
        self.canvas_shape = canvas_shape
        self.image_path = image_path

    def paste_to_canvas(self):
        canvas = np.zeros(self.canvas_shape, np.uint8)
        canvas_height, canvas_width, channels = self.canvas_shape

        image = cv2.imread(self.image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        (H, W) = image.shape[:2]

        plt.imshow(image)
        plt.show()

        ph = canvas_height / float(H)
        h = int(image.shape[0] * ph)
        w = int(image.shape[1] * ph)
        new_image = cv2.resize(image, (w, h))

        plt.imshow(new_image)
        plt.show()

        print(new_image.shape[0], canvas.shape[0])
        print(new_image.shape[1], canvas.shape[1])

        y_off = round((canvas.shape[0] - new_image.shape[0]) / 2)
        x_off = round((canvas.shape[1] - new_image.shape[1]) / 2)
        print(y_off, x_off)

        result = canvas.copy()
        result[y_off:y_off + new_image.shape[0], x_off:x_off + new_image.shape[1]] = new_image

        plt.imshow(result)
        plt.show()
