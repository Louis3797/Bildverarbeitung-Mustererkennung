import cv2
import numpy as np


# Helper functions
def get_int_input(prompt: str) -> int:
    """
    Helper function that is a wrapper around the input function to check if the user entered a int or something else.
    :param prompt: Prompt that the input prints
    :return: Returns the entered integer value of the user
    """
    try:
        return int(input(prompt))
    except ValueError:
        print("Invalid input. Please enter an integer.")
        return get_int_input(prompt)


# Solution function for Task 1:
def fill_image_with_tiles(original_img, tile_img, tile_size: int):
    """
    This function pastes tiles of the specified tile image onto the specified image
    (original image) and returns the new image
    :param original_img: The image we paste the tiles onto
    :param tile_img: The image we choose the tiles from
    :param tile_size: Specified size of the tiles we paste on the original image
    :return: Returns the original image with tiles of the tile image on it
    """

    # Get the dimensions of the original image
    original_height, original_width, _ = original_img.shape

    # Calculate the number of tiles needed in horizontal and vertical directions
    num_horizontal_tiles = original_width // (tile_size + tile_size)
    num_vertical_tiles = original_height // (tile_size + tile_size)

    for y in range(num_vertical_tiles):
        for x in range(num_horizontal_tiles):
            # Choose a random tile patch from the specified tile image
            tile_patch_x = np.random.randint(0, tile_img.shape[1] - tile_size)
            tile_patch_y = np.random.randint(0, tile_img.shape[0] - tile_size)
            tile_patch = tile_img[tile_patch_y:tile_patch_y + tile_size, tile_patch_x:tile_patch_x + tile_size]

            # Calculate the coordinates of the tile in the filled image
            filled_x = x * (tile_size + tile_size)
            filled_y = y * (tile_size + tile_size)

            # Paste the tile patch onto the filled image
            original_img[filled_y:filled_y + tile_size, filled_x:filled_x + tile_size] = tile_patch

    return original_img


# Solution function for Task 2:
def recolor_image(img):
    """
    Recolors greyscale image and generates a new image with the grey values as gradient
    :param img: Given greyscale image
    :return: Returns greyscale gradient image
    """
    # Compute the histogram of the input image
    hist, _ = np.histogram(img, 256, [0, 256])

    max_gray_value = hist[-1]
    curr_pixel_count = 0
    curr_gray_value = 0

    # Create a new image with the same shape as the input image
    recolored_img = np.empty_like(img, dtype=np.uint8)

    height, width = img.shape

    for i in range(height):
        for j in range(width):
            # Find the next gray value that has enough remaining pixels
            while curr_pixel_count >= hist[curr_gray_value] and curr_gray_value < max_gray_value:
                curr_gray_value += 1
                curr_pixel_count = 0

            # Set the current pixel to the next gray value
            recolored_img[i, j] = curr_gray_value
            curr_pixel_count += 1

    return recolored_img


# Load images
minden_img = cv2.imread('p02_minden.jpg')
sun_img = cv2.imread('p02_sonne.jpg')

# Task 1:
tile_size = get_int_input("Please enter the tile size: ")

# Generate image with tiles in it
img_with_tiles = fill_image_with_tiles(minden_img, sun_img, tile_size)
# Display resulting image
cv2.imshow('Tiles', img_with_tiles)

# Task 2:
minden_img2 = cv2.imread('p02_minden.jpg')

# Convert image color to greyscale
grayscale_minden_image = cv2.cvtColor(minden_img2, cv2.COLOR_BGR2GRAY)
# Colorize the image based on the color map
recolored_minden_image = recolor_image(grayscale_minden_image)

# Display resulting image
cv2.imshow('Recolored Image', recolored_minden_image)

# Wait for key press
cv2.waitKey(0)
cv2.destroyAllWindows()
