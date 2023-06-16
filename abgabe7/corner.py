from typing import Any

import cv2


def calculate_eigenvalues(x2, y2, xy) -> tuple[float, float] | None:
    a = 1
    b = -(x2 + y2)
    c = x2 * y2 - xy * xy

    # calculate the discriminant
    discriminant = b * b - 4 * a * c

    if discriminant < 0:
        # No real roots, return None
        return None

    # calculate the eigenvalues using the quadratic formula
    sqrt_discriminant = discriminant ** 0.5
    eigenvalue_1 = (-b + sqrt_discriminant) / (2 * a)
    eigenvalue_2 = (-b - sqrt_discriminant) / (2 * a)

    return eigenvalue_1, eigenvalue_2


def harris_corner_detection(image, k=0.06, threshold=0.01, alpha=1000) -> list[tuple[int, int, tuple[Any, Any]]]:
    # convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # compute the gradients using Sobel operators
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    # compute the elements of the Harris matrix
    sobel_x2 = sobel_x * sobel_x
    sobel_y2 = sobel_y * sobel_y
    sobel_xy = sobel_x * sobel_y

    box_filter_size = 5

    # apply Box filter to the elements of the Harris matrix
    sobel_x2 = cv2.boxFilter(sobel_x2, -1, (box_filter_size, box_filter_size))
    sobel_y2 = cv2.boxFilter(sobel_y2, -1, (box_filter_size, box_filter_size))
    sobel_xy = cv2.boxFilter(sobel_xy, -1, (box_filter_size, box_filter_size))

    # compute the Harris response for each pixel
    det = sobel_x2 * sobel_y2 - sobel_xy * sobel_xy
    trace = sobel_x2 + sobel_y2
    harris_response = det - k * trace * trace

    # Threshold the Harris response
    harris_response[harris_response < threshold * harris_response.max()] = 0

    height, width = harris_response.shape[:2]

    # stores found corners
    # stores the corner as tuple were index 0 is the x coordinate,
    # index 1 is the y coordinate and index 2 is the eigenvalues
    corners = []

    # find the corner coordinates and calculate eigenvalues
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            if harris_response[y, x] != 0:
                x2 = sobel_x2[y, x]
                xy = sobel_xy[y, x]
                y2 = sobel_y2[y, x]

                # get eigenvalues
                e = calculate_eigenvalues(x2, y2, xy)

                # if eigenvalues is not None and min(e[0], e[1]) > alpha
                if e is not None and min(e[0], e[1]) > alpha:
                    corners.append((x, y, (e[0], e[1])))

    return corners


# load image
image = cv2.imread('p07_ecken.png')
corners = harris_corner_detection(image)

# display the corners on the image by print circles where corners are
for corner in corners:
    x, y, eigenvalues = corner
    # draw red circles were a corner is
    cv2.circle(image, (x, y), 1, (0, 0, 255), 2)
    print(f"Corner at (x={x}, y={y}), Eigenvalues: {eigenvalues}")

cv2.imshow('Harris Corner Detection Result', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
