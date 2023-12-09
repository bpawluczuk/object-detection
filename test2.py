import cv2
import numpy as np

# Wczytanie obrazu
image = cv2.imread('images/shelf_2.jpg', cv2.IMREAD_GRAYSCALE)

p = 0.2
w = int(image.shape[1] * p)
h = int(image.shape[0] * p)
image = cv2.resize(image, (w, h))

# Wyrównywanie histogramu
equ_image = cv2.equalizeHist(image)

# Filtracja szumów - zastosowanie filtru Gaussa
gaussian_image = cv2.GaussianBlur(equ_image, (3, 3), 0)

# Normalizacja intensywności pikseli
norm_image = cv2.normalize(gaussian_image, None, 0, 255, cv2.NORM_MINMAX)

# Zastosowanie operatora Sobela
sobel_x = cv2.Sobel(norm_image, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(norm_image, cv2.CV_64F, 0, 1, ksize=3)

# Obliczenie modułu gradientu
gradient = np.sqrt(np.power(sobel_x, 2) + np.power(sobel_y, 2))
gradient = cv2.convertScaleAbs(gradient)

# Progowanie gradientu
threshold = 50
binary_gradient = cv2.threshold(gradient, threshold, 255, cv2.THRESH_BINARY)[1]

# Uzupełnianie krawędzi
edges = cv2.Canny(binary_gradient, 30, 150)

# Rozszerzanie (dilate) krawędzi
kernel = np.ones((3, 3), np.uint8)
dilated_edges = cv2.dilate(edges, kernel, iterations=1)

# Wyświetlenie oryginalnego obrazu oraz wynikowego obrazu z detekcją konturów
cv2.imshow('Original Image', image)
cv2.imshow('Processed Image', binary_gradient)
cv2.imshow('Edges with Edge Completion', dilated_edges)
cv2.waitKey(0)
cv2.destroyAllWindows()




