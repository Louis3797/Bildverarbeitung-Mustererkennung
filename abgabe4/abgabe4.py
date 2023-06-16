import cv2
import numpy as np

original_image_url = "p04_arches.jpg"
sigma = 5.0
N = 3


def apply_laplacian_of_gaussian(image, kernel_size, sigma):
    # Apply Gaussian smoothing
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

    # Apply Laplacian operator
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F)

    return laplacian


def change_brightness(image, brightness_factor):
    # Convert the image to floating-point representation
    image_float = image.astype(np.float32) / 255.0

    # Apply the brightness factor
    image_brightness = image_float * brightness_factor

    # Clip the pixel values to the range [0, 1]
    image_brightness = np.clip(image_brightness, 0, 1)

    # Convert the image back to the original data type
    image_brightness = (image_brightness * 255).astype(np.uint8)

    return image_brightness


# ------------------------ Task 1a ------------------------

def gaussian(x: int, y: int, sigma: float):
    return (1.0 / (2 * np.pi * (sigma ** 2))) * np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))


def create_kernel(n: int, sigma: float) -> np.ndarray:
    filter_size = 2 * n + 1

    # Create empty Kernel
    kernel = np.zeros((filter_size, filter_size), np.float32)

    m = filter_size // 2
    n = filter_size // 2

    # Calculate Values for kernel
    for x in range(-m, m + 1):
        for y in range(-n, n + 1):
            kernel[x + m, y + n] = gaussian(x, y, sigma)

    kernel /= kernel.sum()

    return kernel


# ------------------------ Task 1b ------------------------

def gradient(image):
    # Sobel Kernel x
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])

    # Sobel Kernel y
    sobel_y = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]])

    # Calculate Gradient x and y direction
    gradient_x = cv2.filter2D(image, cv2.CV_64F, sobel_x)
    gradient_y = cv2.filter2D(image, cv2.CV_64F, sobel_y)

    # Calculate Gradient magnitude and Gradient direction
    gradient_magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
    gradient_direction = np.arctan2(gradient_y, gradient_x)

    return gradient_magnitude, gradient_direction


# ------------------------ Task 1c ------------------------

def non_max_suppression(gradient_magnitude, gradient_direction):
    # Create an empty output matrix
    output = np.zeros_like(gradient_magnitude, dtype=np.float32)

    # Round gradient direction to nearest 45 degrees
    direction = np.round(gradient_direction / (np.pi / 4)) % 4

    # Loop over every pixel in the gradient image
    for i in range(1, gradient_magnitude.shape[0] - 1):
        for j in range(1, gradient_magnitude.shape[1] - 1):
            # Check if the current pixel is a local maximum in its direction
            if direction[i, j] == 0:  # Direction: 0 degrees
                output[i, j] = gradient_magnitude[i, j] if (gradient_magnitude[i, j] > gradient_magnitude[
                    i, j - 1]) and (gradient_magnitude[i, j] > gradient_magnitude[i, j + 1]) else 0
            elif direction[i, j] == 1:  # Direction: 45 degrees
                output[i, j] = gradient_magnitude[i, j] if (gradient_magnitude[i, j] > gradient_magnitude[
                    i - 1, j - 1]) and (gradient_magnitude[i, j] > gradient_magnitude[i + 1, j + 1]) else 0
            elif direction[i, j] == 2:  # Direction: 90 degrees
                output[i, j] = gradient_magnitude[i, j] if (gradient_magnitude[i, j] > gradient_magnitude[
                    i - 1, j]) and (gradient_magnitude[i, j] > gradient_magnitude[i + 1, j]) else 0
            else:  # Direction: 135 degrees
                output[i, j] = gradient_magnitude[i, j] if (gradient_magnitude[i, j] > gradient_magnitude[
                    i - 1, j + 1]) and (gradient_magnitude[i, j] > gradient_magnitude[i + 1, j - 1]) else 0

    return output


# ------------------------ Task 1d ------------------------

def hysteresis(gradient_magnitude, low_threshold, high_threshold):
    # Create an output matrix filled with zeros
    output = np.zeros_like(gradient_magnitude)

    # Find the indices of pixels above the high threshold
    strong_i, strong_j = np.where(gradient_magnitude >= high_threshold)

    # Find the indices of pixels between the low and high threshold
    weak_i, weak_j = np.where((gradient_magnitude <= high_threshold) & (gradient_magnitude >= low_threshold))

    # Set strong edges in output
    output[strong_i, strong_j] = 255

    # Check for weak edges that are connected to strong edges
    for i, j in zip(weak_i, weak_j):
        if (i > 0 and output[i - 1, j] == 255) or \
                (i < output.shape[0] - 1 and output[i + 1, j] == 255) or \
                (j > 0 and output[i, j - 1] == 255) or \
                (j < output.shape[1] - 1 and output[i, j + 1] == 255) or \
                (i > 0 and j > 0 and output[i - 1, j - 1] == 255) or \
                (i < output.shape[0] - 1 and j < output.shape[1] - 1 and output[i + 1, j + 1] == 255) or \
                (i > 0 and j < output.shape[1] - 1 and output[i - 1, j + 1] == 255) or \
                (i < output.shape[0] - 1 and j > 0 and output[i + 1, j - 1] == 255):
            output[i, j] = 255

    return output


# ------------------------ Function calls ------------------------

image = cv2.imread(original_image_url)

image_greyscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

kernel = create_kernel(N, sigma)

smoothed_image = cv2.filter2D(image_greyscale, -1, kernel)

gradient_magnitude, gradient_direction = gradient(smoothed_image)
non_max_suppres_image = non_max_suppression(gradient_magnitude,
                                            cv2.filter2D(gradient_direction, -1, create_kernel(3, 5.0)))

result = hysteresis(non_max_suppres_image, 25, 35)

cv2.imshow('Original Image', image_greyscale)
cv2.imshow('Gradient Magnitude Image', gradient_magnitude)
cv2.imshow('Gradient Direction Image', gradient_direction)
cv2.imshow('Gradient Direction Smoothed Image', cv2.filter2D(gradient_direction, -1, create_kernel(3, 5.0)))
cv2.imshow('non_maxima_supression Image', non_max_suppres_image)
cv2.imshow('Hysteresis Image', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
