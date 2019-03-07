import cv2
import numpy as np
from numba import njit


def histogram_stretching(image):
    histogram, _ = np.histogram(image.ravel(), 256, [0, 256])
    return _numba_function(image, histogram)


@njit
def _numba_function(image, histogram):
    min_value = 0
    while histogram[min_value] <= 0:
        min_value += 1

    max_value = 255
    while histogram[max_value] <= 0:
        max_value -= 1

    temp = 255/(max_value - min_value)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            image[i, j] = temp * (image[i, j] - min_value)
    return image


if __name__ == '__main__':
    try:
        img = cv2.imread('hist1.png')
        if img is None:
            raise FileNotFoundError
    except FileNotFoundError:
        print('File not found!')
        exit(0)

    img = histogram_stretching(img)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()