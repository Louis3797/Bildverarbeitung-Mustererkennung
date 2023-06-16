import random

import numpy as np
import cv2


# ------------------------ Task 1 ------------------------

def create_horizontal_linear_filter(n: int):
    if n % 2 == 0:
        raise ValueError("Only odd sizes for the filter are valid")

    # create a 2D numpy array of zeros with size 2*n+1 x 2*n+1
    filter_array = np.zeros((2 * n + 1, 2 * n + 1))

    # set the middle row of the array to ones
    filter_array[n, :] = np.ones(2 * n + 1)

    # normalize the filter so that the sum of its elements equals 1
    filter_array = filter_array / np.sum(filter_array)

    return filter_array


def apply_linear_filter(image, filter: np.ndarray) -> np.ndarray:
    height, width = image.shape
    filter_height, filter_width = filter.shape

    # Calculate the amount of padding needed around the image
    padding_height, padding_width = filter_height // 2, filter_width // 2

    # Pad the image with zeros
    padded_image = np.pad(image, ((padding_height, padding_height), (padding_width, padding_width)), mode='constant',
                          constant_values=0)

    # Create an array to hold the filtered image
    filtered_image = np.zeros_like(image)

    for y in range(height):
        for x in range(width):
            filtered_image[y, x] = np.sum(padded_image[y:y + filter_height, x:x + filter_width] * filter)

    return filtered_image


def add_image_and_mask(image, masked_image):
    height, width = image.shape

    result = np.zeros_like(image)

    for x in range(height):
        for y in range(width):
            if masked_image[x, y] == 0:
                result[x, y] = image[x, y]
            else:
                result[x, y] = masked_image[x, y]

    return result


# ------------------------ Task 2 ------------------------
def weighted_median_filter(image, weights):
    if weights.shape[0] != weights.shape[1] or weights.shape[0] % 2 == 0:
        raise ValueError("Weighted matrix must be square with an odd length")

    image_height, image_width = image.shape

    padding_height, padding_width = weights.shape[0] // 2, weights.shape[1] // 2

    padded_image = np.pad(image, ((padding_height, padding_height), (padding_width, padding_width)), mode='constant')

    # Create an empty resulting image
    result = np.zeros_like(image)

    for i in range(image_height):
        for j in range(image_width):
            # Extract neighborhood of pixel (i, j)
            neighborhood = padded_image[i:i + weights.shape[0], j:j + weights.shape[1]]

            # Calculate weighted median of neighborhood
            weighted_values = np.repeat(neighborhood.flatten(), weights.flatten())
            weighted_median = np.median(weighted_values)
            result[i, j] = weighted_median

    return result


# ------------------------ Task 1 inputs ------------------------
# load images
p03_car = cv2.imread('p03_car.png')
p03_mask = cv2.imread('p03_maske.png', 0)

# to greyscale
car_grey = cv2.cvtColor(p03_car, cv2.COLOR_BGR2GRAY)

# ------------------------ Task 1 B ------------------------
N = 11
# create filter
horizontal_linear_filter = create_horizontal_linear_filter(N)

# apply mask
# Create a binary mask from p03_mask
mask = (p03_mask != 0)
# Create an array of zeros with the same shape as car_grey
masked_image = np.zeros_like(car_grey)
# Only copy the elements of car_grey that satisfy the condition
masked_image[mask] = car_grey[mask]

# apply filter
filtered_img = apply_linear_filter(car_grey, horizontal_linear_filter)

cv2.imshow('Original Image', p03_car)
cv2.imshow('Filtered Image', filtered_img)
cv2.imshow('Masked Image', masked_image)
cv2.imshow('Result 1B Image', add_image_and_mask(filtered_img, masked_image))

# ------------------------ Task 2 ------------------------

# Define the weights for an N=1 filter
weights = np.array([[1, 2, 1],
                    [2, 4, 2],
                    [1, 2, 1]])

median_filter_result = weighted_median_filter(car_grey, weights)
cv2.imshow('Weighted Median Filter Image Task 2', median_filter_result)

cv2.waitKey(0)
cv2.destroyAllWindows()
