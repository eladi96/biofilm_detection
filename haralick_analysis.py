import cv2
import os
import mahotas as mt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


def training_features():
    # load the training dataset
    train_path = os.getcwd() + "/training"

    # empty list to hold feature vectors and train labels
    train_features = []

    for file in os.listdir(train_path):
        # read the training image
        image = cv2.imread(train_path + "/" + file)

        # convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # extract haralick texture from the image
        features = mt.features.haralick(gray)

        # take the mean of its 4 directions
        features = features.mean(axis=0)

        # append the feature vector
        train_features.append(features)

    return train_features


def data_features():
    # load the test dataset
    test_path = os.getcwd() + "/test"

    # empty list to hold feature vectors and train labels
    test_features = dict()

    for file in os.listdir(test_path):
        # read the training image
        image = cv2.imread(test_path + "/" + file)

        # convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # extract haralick texture from the image
        features = mt.features.haralick(gray)

        # take the mean of its 4 angles
        features = features.mean(axis=0)

        # append the feature vector
        test_features[file] = features

    return test_features


# Main Execution
if __name__ == '__main__':
    training_set = training_features()
    test_set = data_features()

    # Normalize the dataset with Standard Scaler
    ss = StandardScaler()
    ss.fit(training_set)
    training_set = ss.transform(training_set)
    scaled_test = ss.transform(list(test_set.values()))
    for file, scaled in zip(test_set.keys(), scaled_test):
        test_set[file] = scaled

    # Train the classifier
    if_clf = IsolationForest()
    if_clf.fit(training_set)

    for file in test_set.keys():
        prediction = if_clf.predict(test_set[file].reshape(1, -1))
        if prediction == 1:
            prediction = 'Biofilm'
        else:
            prediction = 'Other'

        image = cv2.imread(os.getcwd() + "/test/" + file)

        # show the label
        cv2.putText(image, prediction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)

        # display the output image
        cv2.imshow("Test_Image", image)
        cv2.waitKey(0)
