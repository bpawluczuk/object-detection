import cv2

# Wczytaj obraz wejściowy
obraz_wejsciowy = cv2.imread("images/general.jpg")

# Ustal rozmiary komórek siatki w osiach x i y
rozmiar_komorki_x = 300  # Rozmiar komórki w osi x w pikselach
rozmiar_komorki_y = 750  # Rozmiar komórki w osi y w pikselach

# Pętla rysująca siatkę na obrazie
for x in range(0, obraz_wejsciowy.shape[1], rozmiar_komorki_x):
    cv2.line(obraz_wejsciowy, (x, 0), (x, obraz_wejsciowy.shape[0]), (0, 255, 0), 2)

for y in range(0, obraz_wejsciowy.shape[0], rozmiar_komorki_y):
    cv2.line(obraz_wejsciowy, (0, y), (obraz_wejsciowy.shape[1], y), (0, 255, 0), 2)

# Wyświetl obraz z siatką
cv2.imshow("Obraz z siatką", obraz_wejsciowy)
cv2.waitKey(0)
cv2.destroyAllWindows()

