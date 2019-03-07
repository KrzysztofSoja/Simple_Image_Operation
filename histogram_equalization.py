import cv2
import numpy as np
from numba import njit


def histogram_equalization(image):
    histogram, _ = np.histogram(image.ravel(), 256, [0, 256])
    return _numba_function(image, histogram)


@njit
def _numba_function(image, histogram):
    r = image.shape[0] * image.shape[1]
    D = np.zeros((256,))
    for i in range(256):
        for k in range(0, i+1):
            D[i] += histogram[k]
        D[i] /= r

    n = 0
    while D[n] <= 0:
        n += 1
    min_D = D[n]
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            temp = (D[image[i, j]] - min_D)/(1 - min_D)
            image[i, j] = np.floor(255 * temp)
    return image


if __name__ == '__main__':
    try:
        img = cv2.imread('hist1.png', 0)
        if img is None:
            raise FileNotFoundError
    except FileNotFoundError:
        print('File not found!')
        exit(0)

    img = histogram_equalization(img)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
