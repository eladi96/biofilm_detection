import os
import cv2
from scipy.stats import skew, kurtosis
import numpy as np


def hsv_features(image):
    # Convert the image to HSV color model
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Separate the channels
    H, S, V = cv2.split(cv2.cvtColor(hsv_image, cv2.COLOR_BGR2HSV))

    # Equalize the saturation channel
    eq_S = cv2.equalizeHist(S)
    eq_image = cv2.cvtColor(cv2.merge([H, eq_S, V]), cv2.COLOR_HSV2BGR)

    cv2.imshow("eq", eq_image)
    cv2.waitKey()


if __name__ == '__main__':

    path = os.getcwd() + "/images"

    for file in os.listdir(path):
        image = cv2.imread(path + "/" + file)
        hsv_features(image)
