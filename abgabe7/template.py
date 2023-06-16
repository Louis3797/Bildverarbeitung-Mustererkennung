import numpy as np
import cv2

# load images
reference = cv2.imread('p07_reference.png')
template = cv2.imread('p07_template.png')

# convert to greyscale
reference_grey = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
template_grey = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)


def ssd_template_matching(image: np.ndarray, template: np.ndarray) -> tuple[tuple | tuple[int, int], np.ndarray]:
    img_height, img_width = image.shape[:2]
    tpl_height, tpl_width = template.shape[:2]

    # result map
    ssd_map = np.zeros((img_height - tpl_height + 1, img_width - tpl_width + 1))

    # iterate over reference image
    for x in range(ssd_map.shape[0]):
        for y in range(ssd_map.shape[1]):
            region = image[x:x + tpl_height, y:y + tpl_width]
            diff = np.sum((region - template) ** 2)
            ssd_map[x, y] = diff

    min_location = np.unravel_index(np.argmin(ssd_map), ssd_map.shape)

    return min_location, ssd_map


def cor_template_matching(image: np.ndarray, template: np.ndarray) -> tuple[tuple | tuple[int, int], np.ndarray]:
    img_height, img_width = image.shape[:2]
    tpl_height, tpl_width = template.shape[:2]

    # result map
    cor_map = np.zeros((img_height - tpl_height + 1, img_width - tpl_width + 1))

    # iterate over image
    for i in range(cor_map.shape[0]):
        for j in range(cor_map.shape[1]):
            region = image[i:i + tpl_height, j:j + tpl_width]

            # Subtract the means of the region and template
            region_mean = np.mean(region)
            template_mean = np.mean(template)
            region_diff = region - region_mean
            template_diff = template - template_mean

            # Calculate the correlation coefficient
            numerator = np.sum(region_diff * template_diff)
            region_std = np.sqrt(np.sum(region_diff ** 2))
            template_std = np.sqrt(np.sum(template_diff ** 2))

            # add to map
            cor_map[i, j] = numerator / (region_std * template_std)

    # find the location with the correlation coefficient
    max_location = np.unravel_index(np.argmax(cor_map), cor_map.shape)

    return max_location, cor_map


def visualize_results(reference: np.ndarray, template: np.ndarray, coordinates: tuple[int, int]) -> np.ndarray:
    tpl_height, tpl_width = template.shape[:2]

    # copy reference image
    result = np.copy(reference)

    # get coordinates
    x, y = coordinates

    # draw rectangle to visualize the result
    cv2.rectangle(result, (y - 1, x - 1), (y + tpl_width + 1, x + tpl_height + 1), (0, 0, 255), 2)

    return result


ssd_coordinates, ssd_result_map = ssd_template_matching(reference_grey, template_grey)
cor_coordinates, cor_result_map = cor_template_matching(reference_grey, template_grey)

# visualize the results
ssd_result = visualize_results(reference, template, ssd_coordinates)
cor_result = visualize_results(reference, template, cor_coordinates)

cv2.imshow('Reference', reference)
cv2.imshow('Template', template)
cv2.imshow('SSD Map', ssd_result_map)
cv2.imshow('COR Map', cor_result_map)
cv2.imshow('SSD Result', ssd_result)
cv2.imshow('COR Result', cor_result)
cv2.waitKey(0)
cv2.destroyAllWindows()
