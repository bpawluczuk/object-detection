import tensorflow as tf
import numpy as np
import cv2

# Wczytaj model Mask R-CNN
model = tf.keras.models.load_model("mask_rcnn.h5", compile=False)

# Wczytaj obraz
image = cv2.imread("images/shelf_1.jpg")

# Przekształć obraz na wymiar o wielkości akceptowanej przez model
image = cv2.resize(image, (800, 800))

# Wykonaj detekcję obiektów i segmentację
result = model.detect([image])

# Wyodrębnij wyniki z detekcji i segmentacji
r = result[0]
masks = r["masks"]
class_ids = r["class_ids"]

# Wyodrębnij kolorową mapę obiektów
colors = np.random.randint(0, 255, size=(len(class_ids), 3), dtype=np.uint8)

# Utwórz pustą maskę
mask = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

# Wykonaj segmentację dla każdej instancji obiektu
for i in range(masks.shape[2]):
    # Wyodrębnij maskę dla instancji obiektu
    mask_i = masks[:, :, i]

    # Przypisz unikalny kolor dla instancji obiektu
    color = colors[i]

    # Przypisz kolor maskom dla instancji obiektu
    mask[:, :, 0] = np.where(mask_i == 1, color[0], mask[:, :, 0])
    mask[:, :, 1] = np.where(mask_i == 1, color[1], mask[:, :, 1])
    mask[:, :, 2] = np.where(mask_i == 1, color[2], mask[:, :, 2])

# Wyświetl wynik
cv2.imshow("Panoptic Segmentation", mask)
cv2.waitKey(0)


