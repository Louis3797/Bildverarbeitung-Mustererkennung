import cv2
import numpy as np
import sys

np.set_printoptions(threshold=sys.maxsize)


def dilate(image, structure_element, iterations=1):
    output = np.copy(image)
    k_h, k_w = structure_element.shape
    pad_h = k_h // 2
    pad_w = k_w // 2

    for _ in range(iterations):
        padded_image = np.pad(output, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
        dilated = np.zeros_like(output)
        for i in range(k_h):
            for j in range(k_w):
                if structure_element[i, j] != 0:
                    dilated = np.maximum(dilated, padded_image[i:i + output.shape[0], j:j + output.shape[1]])
        output = dilated

    return output


def erode(image, structure_element, iterations=1):
    output = np.copy(image)
    k_h, k_w = structure_element.shape
    pad_h = k_h // 2
    pad_w = k_w // 2

    for _ in range(iterations):
        padded_image = np.pad(output, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
        eroded = np.ones_like(output) * 255
        for i in range(k_h):
            for j in range(k_w):
                if structure_element[i, j] != 0:
                    eroded = np.minimum(eroded, padded_image[i:i + output.shape[0], j:j + output.shape[1]])
        output = eroded

    return output


def closing_operation(image, structure_element):
    dilated = dilate(image, structure_element, 1)
    closed = erode(dilated, structure_element, 1)

    return closed


original_image = cv2.imread('p06_zahnrad.png', cv2.IMREAD_GRAYSCALE)

N = 5
kernel = np.ones((N, N), dtype=np.uint8)

cv2.imshow('Original Image', original_image)
cv2.imshow('Dilation Operation Image', dilate(original_image, kernel))
cv2.imshow('Erosion Operation Image', erode(original_image, kernel))
cv2.imshow('Closing Operation Image', closing_operation(original_image, kernel))

cv2.waitKey(0)
cv2.destroyAllWindows()
