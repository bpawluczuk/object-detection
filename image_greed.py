import cv2

# wczytanie obrazu
img = cv2.imread('images/shelf_3.jpg')

# podzielenie obrazu na siatkę
szerokosc, wysokosc, _ = img.shape
szerokosc_siatki = 100
wysokosc_siatki = 100

# sprawdzenie, które obszary siatki zawierają obiekt
obszary = []
for x in range(0, szerokosc, szerokosc_siatki):
    for y in range(0, wysokosc, wysokosc_siatki):
        roi = img[y:y + wysokosc_siatki, x:x + szerokosc_siatki]
        # tu można dodać kod, który sprawdza czy roi zawiera obiekt
        # np. użyć modelu detekcji obiektów
        # jeśli roi zawiera obiekt, to dodajemy go do listy obszarów
        if 1:
            obszary.append((x, y, szerokosc_siatki, wysokosc_siatki))

# narysowanie prostokątów dla obszarów zawierających obiekt
for x, y, szerokosc, wysokosc in obszary:
    cv2.rectangle(img, (x, y), (x + szerokosc, y + wysokosc), (0, 255, 0), 2)

# wyświetlenie obrazu z zaznaczonymi obszarami zawierającymi obiekt
cv2.imshow('Obraz z zaznaczonymi obszarami', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
