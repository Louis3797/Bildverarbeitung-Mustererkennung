import cv2
import numpy as np

# min vote threshold
threshold = 135


def hough_circles(image, radius, hough_threshold) -> tuple[int, np.ndarray]:
    height, width = image.shape[:2]

    accumulator = np.zeros(image.shape)

    # blur and smooth image with gaussian kernel
    blurred_img = cv2.GaussianBlur(image, (5, 5), 1)

    # get edges with canny edge detection
    edges = cv2.Canny(blurred_img, 150, 200)

    # iterate over the image
    for x in range(height - radius):
        for y in range(width - radius):
            # check if pixel is an edge pixel
            if edges[x][y] > 0:
                for theta in range(0, 360):
                    angle = np.deg2rad(theta)
                    # calculate polar coordinate points
                    a = int(x - radius * np.cos(angle))
                    b = int(y - radius * np.sin(angle))

                    accumulator[a, b] += 1

    # get all center candidates from the accumulator matrix
    # meaning all points where the vote value exceeds the set threshold
    center_candidates = np.where(accumulator > hough_threshold)

    # image for the circles
    circles = np.zeros(image.shape)

    # counter for the found coins
    count = 0

    # iterate through the center candidates
    for i in range(0, center_candidates[0].size):
        flag = True
        for theta in range(0, 360):
            angle = np.deg2rad(theta)
            # calculate polar coordinate points
            y = center_candidates[1][i] - int(radius / 10) * np.sin(angle)
            x = center_candidates[0][i] - int(radius / 10) * np.cos(angle)

            # check if there is already a different center in between the current center and the edge
            if x < width and y < height:
                if np.sum(circles[center_candidates[0][i]:int(x), center_candidates[1][i]:int(y)]) > 128:
                    flag = False
        if flag:
            # increment coin count
            count += 1
            # draw center of coins
            cv2.circle(circles, (center_candidates[1][i], center_candidates[0][i]), 2, 255, 2)

            # draw circle around the coins
            cv2.circle(circles, (center_candidates[1][i], center_candidates[0][i]), radius, 128, 2)

    return count, circles


def color_coins(image: np.ndarray, circle_1, circle_2, circle_3, threshold=128):
    # convert grayscale image to RGB image for drawing colored circles
    output = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # draw circle for 1 cent red
    output[circle_1 >= threshold] = [0, 0, 255]  # Red
    # draw circle for 2 cent cyan
    output[circle_2 >= threshold] = [255, 255, 0]  # Cyan
    # draw circle for 5 cent blue
    output[circle_3 >= threshold] = [255, 0, 0]  # Blue

    return output


image = cv2.imread("p08_muenzen.png", cv2.IMREAD_GRAYSCALE)

# apply Hough circle transformation to find the circles and coin counts
one_cent_coin_count, one_cent_circles = hough_circles(image, 23, threshold)
two_cent_coin_count, two_cent_circles = hough_circles(image, 27, threshold)
five_cent_coin_count, five_cent_circles = hough_circles(image, 32, threshold)

# get result image
result = color_coins(image, one_cent_circles, two_cent_circles, five_cent_circles)

# calculate total of coins
coin_total = one_cent_coin_count + two_cent_coin_count * 2 + five_cent_coin_count * 5

# print coin count
print(f"1 cent coins: {one_cent_coin_count}")
print(f"2 cent coins: {two_cent_coin_count}")
print(f"5 cent coins: {five_cent_coin_count}")

# calculate sum
print(f"The total is: {coin_total} cent")

# display results
cv2.imshow("Circles 1 cent", one_cent_circles)
cv2.imshow("Circles 2 cent", two_cent_circles)
cv2.imshow("Circles 5 cent", five_cent_circles)
cv2.imshow("Hough circle result", result)

cv2.waitKey()
cv2.destroyAllWindows()
