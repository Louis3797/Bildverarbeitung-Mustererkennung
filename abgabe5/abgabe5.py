import numpy as np
import cv2


def otsu_thresholding(image):
    # Get histogram
    hist, _ = np.histogram(image.flatten(), bins=256, range=[0, 256])
    hist = hist.astype(float)

    hist /= hist.sum()  # normalize histogram

    total_pixels = image.size
    current_max_variance = 0.0
    threshold = 0

    # Durchlaufe alle Schwellenwerte von 0 bis 255
    for t in range(256):

        # Berechne die Wahrscheinlichkeiten der beiden Klassen
        w0 = hist[:t].sum()
        w1 = hist[t:].sum()

        # Berechne die mittleren Intensitäten der beiden Klassen
        if w0 == 0 or w1 == 0:
            continue

        mu0 = np.dot(np.arange(t), hist[:t]) / w0
        mu1 = np.dot(np.arange(t, 256), hist[t:]) / w1

        # Berechne die Varianz zwischen den Klassen
        variance = w0 * w1 * (mu0 - mu1) ** 2

        # Aktualisiere den maximalen Varianzwert und den Schwellenwert
        if variance > current_max_variance:
            current_max_variance = variance
            threshold = t

    # Wende den Schwellenwert auf das Bild an
    binary_image = (image < threshold).astype(np.uint8) * 255

    return binary_image


image = cv2.imread("p05_gummibaeren.png")

segmented_img = otsu_thresholding(image)

cv2.imshow('Hysteresis Image', segmented_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
