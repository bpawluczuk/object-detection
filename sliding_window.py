from joblib.numpy_pickle_utils import xrange
import cv2


def sliding_window(image, window_size, step_size):
    for y in xrange(0, image.shape[0], step_size[1]):
        for x in xrange(0, image.shape[1], step_size[0]):
            yield x, y, image[y: y + window_size[1], x: x + window_size[0]]


original_image = cv2.imread("images/shower_all.jpg")
print("oryginal: ", original_image.shape[0:2])
min_wdw_sz = (150, 500)
step_size = (50, 500)
downscale = 1

print("resize: ", original_image.shape[0:2])

for (x, y, im_window) in sliding_window(original_image, min_wdw_sz, step_size):
    print(im_window.shape[0:2])
    cv2.imshow("Output", im_window)
    key = cv2.waitKey(0) & 0xFF

print("end")
