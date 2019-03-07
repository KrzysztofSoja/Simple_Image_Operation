import cv2
import numpy as np
from numba import njit


def bersen_thresholding(image, epsilon=25):
    threshold, _ = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return _numba_function(threshold, image, epsilon)


@njit
def _numba_function(global_threshold, image, epsilon):
    radius = 3
    new_image = np.zeros_like(image)
    for i in range(radius, image.shape[0]-radius):
        for j in range(radius, image.shape[1]-radius):
            surroundings = np.zeros((2 * radius - 1, 2 * radius - 1))
            for i_s in range(surroundings.shape[0]):
                for j_s in range(surroundings.shape[1]):
                    surroundings[i_s][j_s] = image[i + i_s - radius][j + j_s - radius]
            threshold = global_threshold
            if np.max(surroundings) - np.min(surroundings) >= epsilon:
                threshold = (np.min(surroundings) + np.max(surroundings))/2
            if threshold > image[i][j]:
                new_image[i, j] = 0
            else:
                new_image[i, j] = 255
    return new_image


if __name__ == '__main__':
    try:
        img = cv2.imread('text.png', 0)
        if img is None:
            raise FileNotFoundError
    except FileNotFoundError:
        print('File not found!')
        exit(0)

    img = bersen_thresholding(img)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()