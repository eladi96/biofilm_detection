import cv2
import numpy as np
import dominant_color as dc
import os
import glob

for filename in glob.glob(os.path.join(os.getcwd() + "/tiles/", '*.jpg')):
    # construct the argument parser and parse the arguments
    args = dict()
    args['imagePath'] = filename
    args['clusters'] = 2

    # read in image of interest
    bgr_image = cv2.imread(args['imagePath'])
    # convert to HSV; this is a better representation of how we see color
    hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)

    # extract dominant color
    # (aka the centroid of the most popular k means cluster)
    dom_color = dc.get_dominant_color(hsv_image, k=args['clusters'])

    # create a square showing dominant color of equal size to input image
    dom_color_hsv = np.full(bgr_image.shape, dom_color, dtype='uint8')
    # convert to bgr color space for display
    dom_color_bgr = cv2.cvtColor(dom_color_hsv, cv2.COLOR_HSV2BGR)

    # concat input image and dom color square side by side for display
    output_image = np.hstack((bgr_image, dom_color_bgr))

    print(dom_color)
    # show results to screen
    cv2.imshow('Image Dominant Color', output_image)
    cv2.waitKey(0)
