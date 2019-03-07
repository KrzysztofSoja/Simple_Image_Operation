import cv2
import os

from histogram_stretching import histogram_stretching
from histogram_equalization import histogram_equalization
from bersen_thersholding import bersen_thresholding
from timer import Timer


def read_image(file_name, layer=0):
    try:
        image = cv2.imread(file_name, layer)
        if image is None:
            raise FileNotFoundError
        return image
    except FileNotFoundError:
        print('File not found!')
        exit(0)


bird_left = read_image('contrast_left.jpg')
bird_right = read_image('contrast_right.jpg')

bird_left = histogram_equalization(bird_left)
bird_right = histogram_equalization(bird_right)

cv2.imwrite(os.getcwd() + '/answer/bird_left.jpg', bird_left)
cv2.imwrite(os.getcwd() + '/answer/bird_right.jpg', bird_right)


people = read_image('kobieta.jpg')
people = histogram_equalization(people)
cv2.imwrite(os.getcwd() + '/answer/kobieta.jpg', people)

text = read_image('text.png')
text = bersen_thresholding(text)
cv2.imwrite(os.getcwd() + '/answer/text.jpg', text)

big = read_image('hist3.jpg')
with Timer() as t:
    big = bersen_thresholding(big)
cv2.imwrite(os.getcwd() + '/answer/big.jpg', big)
print("Time:" + str(t.interval))



