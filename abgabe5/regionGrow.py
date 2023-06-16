import cv2
import numpy as np

seed_pixel = None
threshold = 37

marker_size = 2

def mouse_callback(event, x, y, flags, param):
    global seed_pixel

    if event == cv2.EVENT_LBUTTONDOWN:
        seed_pixel = (x, y)

        # Perform region growing and update the result image
        region_growing_result = region_growing(seed_pixel,  image_bw)

        # Draw a green marker on the result image at the seed pixel location
        region_growing_result[y - marker_size:y + marker_size, x - marker_size:x + marker_size] = (0, 255, 0)
        # Show the region growing image
        cv2.imshow('Region Growing Image', region_growing_result)

    if event == cv2.EVENT_RBUTTONDOWN:
        # Show the original grayscale image when right mouse button is clicked
        cv2.imshow('Region Growing Image', image_bw)


def region_growing(seed_px, image):
    height, width = image.shape[:2]
    seed_value = float(image[seed_px[1], seed_px[0]])

    print(seed_value)

    result_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    result_image[:, :] = (200, 0, 0)

    queue = [seed_px]
    processed_pixels = set()  # Keep track of already processed pixels to avoid duplicates

    while len(queue) > 0:
        current_pixel = queue.pop(0)

        # Skip if the region color is already red (RGB: 0, 0, 255)
        if np.array_equal(result_image[current_pixel[1], current_pixel[0]], [0, 0, 255]):
            continue

        current_value = float(image[current_pixel[1], current_pixel[0]])
        # print(current_value)
        if abs(current_value - seed_value) <= threshold:
            # Set found region color to red (RGB: 0, 0, 255)
            result_image[current_pixel[1], current_pixel[0]] = [0, 0, 255]

            # Add neighboring pixels to the queue for further processing
            for i in range(-1, 2):
                for j in range(-1, 2):
                    if 0 <= current_pixel[1] + i < height and 0 <= current_pixel[0] + j < width:
                        neighbor_pixel = (current_pixel[0] + j, current_pixel[1] + i)
                        if neighbor_pixel not in processed_pixels:
                            queue.append(neighbor_pixel)
                            processed_pixels.add(neighbor_pixel)

    return result_image


image_bw = cv2.imread('p05_gummibaeren.png', cv2.IMREAD_GRAYSCALE)

cv2.namedWindow('Region Growing Image')
cv2.setMouseCallback('Region Growing Image', mouse_callback)
cv2.imshow('Region Growing Image', image_bw)

cv2.waitKey(0)
cv2.destroyAllWindows()
