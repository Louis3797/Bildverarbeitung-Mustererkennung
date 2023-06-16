import cv2
import numpy as np


# 1
def generate_avg_column_image(image) -> np.ndarray:
    """
    Generates a new image by calculating the average of the columns from the specified image
    :param image: Path to the file
    :return: returns None
    """

    # get height and widht properties of the image
    height, width, channels = image.shape

    # create empty output image
    output_image = np.zeros((height, width, channels), dtype=np.uint8)

    # iterate through each col
    for x in range(width):
        # calculate average for col
        mean_color = np.mean(image[:, x, :], axis=0)
        # set average
        output_image[:, x, :] = mean_color

    return output_image


# 2
def rgb_to_yuv(image) -> tuple[float, float, float]:
    """
    Convert an RGB image to YUV color space.
    :param image: Specified image
    :return: Returns a tuple containing three numpy arrays representing the Y, U, and V channels of the YUV image.
    """
    img_prime = np.power(image / 255.0, 1.0 / 2.2)

    # Get RGB channels
    r, g, b = img_prime[:, :, 2], img_prime[:, :, 1], img_prime[:, :, 0]

    y = 0.299 * r + 0.587 * g + 0.114 * b
    u = 0.493 * (b - y)
    v = 0.877 * (r - y)

    return y, u, v


def extract_shadow(image):
    """
    Extracts the shadow of a given YUV color space image
    :param img: Specified image in YUV color space
    :return: Returns an Image with the extracted shadow in white
    """
    # convert rgb to yuv
    _, U, V = rgb_to_yuv(image)

    mu_u, std_u = np.mean(U), np.std(U)
    mu_v, std_v = np.mean(V), np.std(V)

    T1 = np.where(U > mu_u + std_u, 255, 0).astype(np.uint16)
    T2 = np.where(V > mu_v - std_v, 0, 255).astype(np.uint16)

    # Combine T1 and T2 to obtain the shadow region S
    S = T1 * T2

    return S


# Read original image
img = cv2.imread('p01_schatten.jpg')

task1_image = generate_avg_column_image(img)

# Convert image to yuv color image
y, u, v = rgb_to_yuv(img)
yuv_img = np.stack([y, u, v], axis=2)

# Extract the shadow component of the original image
shadow_img = extract_shadow(img)

# Output the result using OpenCV
cv2.imshow('Original Image', img)
cv2.imshow('Task 1 Image', task1_image)
cv2.imshow('YUV Image', yuv_img)
cv2.imshow('Shadow Extraction', shadow_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
