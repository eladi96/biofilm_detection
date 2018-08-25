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

    # Calculate the probability density function histogram for H and S channels
    H_hist = np.histogram(H, bins=180, range=[0, 180], density=True)
    S_hist = np.histogram(eq_S, bins=256, range=[0, 256], density=True)

    print(H_hist.mean())


if __name__ == '__main__':

    path = os.getcwd() + "/training"

    for file in os.listdir(path):
        image = cv2.imread(path + "/" + file)
        hsv_features(image)
