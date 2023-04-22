import cv2

# Wczytanie obrazu czarno-białego
image = cv2.imread('images/shelf_3.jpg', 0)  # 0 oznacza wczytanie obrazu w skali szaro-szarej (grayscale)

# Zastosowanie algorytmu Canny do detekcji krawędzi
edges = cv2.Canny(image, 100, 200)  # 100 i 200 to progi detekcji krawędzi

# Znalezienie konturów na podstawie krawędzi
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Iteracja po konturach i wyznaczenie bounding boxów
bounding_boxes = []
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    bounding_boxes.append((x, y, w, h))

# Wyświetlenie obrazu z zaznaczonymi bounding boxami
image_with_boxes = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # Konwersja obrazu na obraz kolorowy
for bbox in bounding_boxes:
    x, y, w, h = bbox
    cv2.rectangle(image_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Rysowanie bounding boxa na obrazie

cv2.imshow('Obraz z bounding boxami', image_with_boxes)
cv2.waitKey(0)
cv2.destroyAllWindows()




