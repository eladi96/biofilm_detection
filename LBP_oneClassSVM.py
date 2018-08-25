# import the necessary packages
from LBP_analysis import LocalBinaryPatterns
from sklearn.svm import OneClassSVM
import cv2
import os


def apply_overlay(image, x, y, step):
    overlay = image.copy()

    # select the region that has to be overlaid
    cv2.rectangle(overlay, (x, y), (x + step, y + step), (0, 255, 0), -1)

    # Adding the transparency parameter
    alpha = 0.5

    # Performing image overlay
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)


def detect_biofilm(image, desc, model):
    step = 128
    width, height = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    for y in range(0, height, step):
        for x in range(0, width, step):
            tile = gray[y:y + step, x:x + step]
            hist = desc.describe(tile)
            prediction = model.predict(hist.reshape(1, -1))
            if prediction == 1:
                apply_overlay(image, x, y, step)
                cv2.imshow("image", image)
                cv2.waitKey(0)


def train_svm():
    # initialize the local binary patterns descriptor along with
    # the data and label lists
    desc = LocalBinaryPatterns(24, 8)
    data = []
    labels = []

    # loop over the training biofilm
    biofilmPath = os.getcwd() + "/training"
    for imagePath in os.listdir(biofilmPath):
        # load the image, convert it to grayscale, and describe it
        image = cv2.imread(biofilmPath + "/" + imagePath, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hist = desc.describe(gray)

        # extract the label from the image path, then update the
        # label and data lists
        labels.append("biofilm")
        data.append(hist)

    # train a Linear SVM on the data
    model = OneClassSVM()
    model.fit(data)
    return desc, model


# Main execution
if __name__ == "__main__":

    desc, model = train_svm()

    # loop over the testing images
    testPath = os.getcwd() + "/test"

    for imagePath in os.listdir(testPath):
        image = cv2.imread(testPath + "/" + imagePath)
        detect_biofilm(image, desc, model)
