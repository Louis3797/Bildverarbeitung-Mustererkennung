import cv2
import numpy as np

region_growing_threshold = 0  # the image we apply region growing to is black and white

seed = (315, 181)


def apply_region_growing(seed, image, threshold):
    """
    Executes region growing on a given seed position
    :param seed: Start position
    :param image: Image we use region growing on
    :param threshold: Threshold between the colors
    :return:
    """
    height, width = image.shape[:2]
    seed_value = float(image[seed[1], seed[0]])

    result_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    result_image[:, :] = (0, 0, 0)

    queue = [seed]
    processed_pixels = set()  # Keep track of already processed pixels to avoid duplicates

    while len(queue) > 0:
        current_pixel = queue.pop(0)

        # Skip if the region color is already red (RGB: 0, 0, 255)
        if np.array_equal(result_image[current_pixel[1], current_pixel[0]], [255, 255, 255]):
            continue

        current_value = float(image[current_pixel[1], current_pixel[0]])
        # print(current_value)
        if abs(current_value - seed_value) <= threshold:
            # Set found region color to red (RGB: 0, 0, 255)
            result_image[current_pixel[1], current_pixel[0]] = [255, 255, 255]

            # Add neighboring pixels to the queue for further processing
            for i in range(-1, 2):
                for j in range(-1, 2):
                    if 0 <= current_pixel[1] + i < height and 0 <= current_pixel[0] + j < width:
                        neighbor_pixel = (current_pixel[0] + j, current_pixel[1] + i)
                        if neighbor_pixel not in processed_pixels:
                            queue.append(neighbor_pixel)
                            processed_pixels.add(neighbor_pixel)

    return result_image


def thinning(image):
    """
    Skeletonizes structures in image
    :param image: Specified image
    :return: Returns skeletonized image
    """
    input_size = np.size(image)
    thinned_line = np.zeros(image.shape, np.uint8)

    thresholded_img = np.zeros_like(image)
    thresholded_img[image > 0] = 255

    element = np.array([[0, 1, 0],
                        [1, 1, 1],
                        [0, 1, 0]], dtype=np.uint8)

    done = False
    while not done:
        eroded = cv2.erode(thresholded_img, element)
        temp = cv2.dilate(eroded, element)
        temp = np.subtract(thresholded_img, temp)
        thinned_line = np.bitwise_or(thinned_line, temp)
        thresholded_img = eroded.copy()

        zeros = input_size - np.count_nonzero(thresholded_img)
        if zeros == input_size:
            done = True

    kernel = np.ones((5, 5), np.uint8)
    thinned_line = cv2.morphologyEx(thinned_line, cv2.MORPH_CLOSE, kernel)

    return thinned_line


# Load the image
image = cv2.imread('p06_strasse.jpg', cv2.COLOR_BGR2GRAY)  # Read as grayscale

# First we blur the image with a gaussian kernel to remove some noise
image_blurred = cv2.GaussianBlur(image, (9, 9), 0)

# With the blured image we can now apply canny edge detection to find the edges
edges = cv2.Canny(image_blurred, 70, 180)

# Next we dilate the edges to make them stronger
kernel2 = np.ones((5, 5), np.uint8)
dilate = cv2.dilate(edges, kernel2, iterations=1)

# After to dilate we make a closing operation to remove the last noise
closing = cv2.morphologyEx(dilate, cv2.MORPH_CLOSE, kernel2)

# We now apply region growing on the street with a seed that we got before
street_region = apply_region_growing(seed, closing, region_growing_threshold)

street_region_dilate = cv2.dilate(street_region, kernel2, iterations=4)

kernel = np.ones((3, 3), np.uint8)
street_region_dilate = cv2.morphologyEx(street_region_dilate, cv2.MORPH_OPEN, kernel)

# make region red
street_region_dilate[np.where((street_region_dilate == [255, 255, 255]).all(axis=2))] = [0, 0, 255]

# skeletonize
street_line = thinning(street_region_dilate)

# apply line to original image
result = np.bitwise_or(image, street_line)

# Display the original image and edges
cv2.imshow('Original Image', image)
cv2.imshow('Blur Image', image_blurred)
cv2.imshow('Edges', edges)
cv2.imshow('Dilate', dilate)
cv2.imshow('Closing', closing)
cv2.imshow('Street Region', street_region)
cv2.imshow('Street Region Dilate', street_region_dilate)
cv2.imshow('Street Line', street_line)
cv2.imshow('Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
