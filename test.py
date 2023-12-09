import cv2
import numpy as np

# Wczytanie obrazu
obraz_bgr = cv2.imread("images/shelf_2.jpg")

# Konwersja obrazu na przestrzeń kolorów HSV
obraz_hsv = cv2.cvtColor(obraz_bgr, cv2.COLOR_BGR2HSV)

# Zmiana jasności w przestrzeni HSV
# factor = 70  # Faktor zmiany jasności
# obraz_hsv[:, :, 2] = cv2.add(obraz_hsv[:, :, 2], factor)

# Wyodrębnienie komponenty H (ton) z przekonwertowanego obrazu
ton = obraz_hsv[:, :, 0]
# Wyodrębnienie komponenty S (nasycenie) z przekonwertowanego obrazu
nasycenie = obraz_hsv[:, :, 1]
# Wyodrębnienie komponenty V (jasność) z przekonwertowanego obrazu
jasnosc = obraz_hsv[:, :, 2]

# Wykonanie dalszych operacji na komponentach obrazu HSV
# np. progowanie, detekcja konturów itp.

# Wyświetlenie oryginalnego obrazu BGR
cv2.imshow("Oryginalny obraz BGR", obraz_bgr)
obraz_rozjasniony = cv2.cvtColor(obraz_hsv, cv2.COLOR_HSV2BGR)
cv2.imshow("Oryginalny obraz BGR obraz_rozjasniony", obraz_rozjasniony)

# Wyświetlenie przekonwertowanego obrazu HSV
# cv2.imshow("Przekonwertowany obraz HSV", obraz_hsv)
# cv2.imshow("Ton HSV", ton)
# cv2.imshow("Nasycenie HSV", nasycenie)

# jasnosc = cv2.GaussianBlur(jasnosc, (0, 0), 10)
cv2.imshow("Jasność HSV", jasnosc)

equalized_image = cv2.equalizeHist(jasnosc)
cv2.imshow("equalized_image", equalized_image)

ksize = 17  # Rozmiar okna filtru medianowego
denoised_image = cv2.medianBlur(equalized_image, ksize)
cv2.imshow("denoised_image", equalized_image)

blurred = cv2.GaussianBlur(denoised_image, (5, 5), 0)
canny = cv2.Canny(blurred, 10, 70)
cv2.imshow("cany", canny)

# Oczekiwanie na wciśnięcie dowolnego klawisza
cv2.waitKey(0)

# Zamknięcie okien
cv2.destroyAllWindows()

cv2.imwrite('hsv/hsv.jpg', equalized_image)







