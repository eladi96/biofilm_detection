import os
import cv2
from skimage import io

for file in os.listdir(os.getcwd() + "/test"):
    print(file)
    image = cv2.cvtColor(cv2.imread(os.getcwd() + "/test/" + file), cv2.COLOR_BGR2RGB)
    width, height = image.shape[:2]
    step = 128
    c = 0
    for y in range(0, height, step):
        for x in range(0, width, step):
            crop = image[y:y + step, x:x + step]
            io.imsave(os.getcwd() + "/tiles/" + str(c) + "-" + file, crop)
            c += 1
