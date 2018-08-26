import cv2
import os
import mahotas as mt
import numpy as np
import dominant_color as dc
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from collections import Counter


def haralick_features(path, filenames):
    """ Returns a dict containing 13 Haralick texture features for
    every image in the list of files given as input.

    Every image is firs converted to grayscale, only then Haralick features
    are extracted from the four 2D-directions, and the mean of the four directions values
    is calculated.

    Parameters
    ----------
    path : str
        The path which contains the files.
    filenames : list
        The list of files to elaborate.

    Returns
    -------
    features: dict of (str : list of np.double)
        The dictionary containing the list of features for every file.
    """
    features_list = dict()

    for file in filenames:
        image = cv2.imread(path + "/" + file)
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        features = mt.features.haralick(grayscale_image)
        features = features.mean(axis=0)
        features_list[file] = features

    return features_list


def get_dominant_color(image, k=3):
    """Takes an image as input and returns the dominant color of the image as a list.

    Dominant color is found by running K-means on the
    pixels and returning the centroid of the largest cluster.

    Parameters
    ----------
    image : np.array-like image
        The image to elaborate.
    k : int, optional (default = 4)
        The number of clusters to form as well as the number of
        centroids to generate.

    Returns
    -------
    dominant_color : the h, s and v components of the dominant color
        in list form.
    """
    # reshape the image to be a list of pixels
    image = image.reshape((image.shape[0] * image.shape[1], 3))

    # cluster and assign labels to the pixels
    clt = KMeans(n_clusters=k)
    labels = clt.fit_predict(image)

    # count labels to find most popular
    label_counts = Counter(labels)

    # subset out most popular centroid
    dominant_color = clt.cluster_centers_[label_counts.most_common(1)[0][0]]

    return list(dominant_color)


def hsv_features(path, filenames, clusters=3):
    """Returns a list containing the HSV values of the dominant color of every image
    given as input.

    Parameters
    ----------
    path : str
        The path which contains the files.
    filenames : list
        The list of files to elaborate.
    clusters : int, optional (default = 3)
        The number of clusters to use for k-means elaboration.

    Returns
    -------
    dom_color_list : dict of (str : list of np.double)
        The dictionary containing the dominant color for every file.
    """
    dom_color_list = dict()

    for file in filenames:
        bgr_image = cv2.imread(path + "/" + file)
        hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
        dom_color = dc.get_dominant_color(hsv_image, k=clusters)
        dom_color_list[file] = dom_color

    return dom_color_list


if __name__ == '__main__':

    # Definition of paths
    training_path = os.getcwd() + "/training"
    test_path = os.getcwd() + "/test"

    # Create the list of the training and test files
    training_filenames = os.listdir(training_path)
    test_filenames = os.listdir(test_path)

    # Extract haralick features from both lists
    print("Extracting haralick features...")
    haralick_training = haralick_features(training_path, training_filenames)
    haralick_test = haralick_features(test_path, test_filenames)
    print("Completed!")

    # Extract color features from both lists
    print("Extracting color features...")
    hsv_training = hsv_features(training_path, training_filenames)
    hsv_test = hsv_features(test_path, test_filenames)
    print("Completed!")

    # Concatenate the lists to create a single feature vector per file
    training_set = dict()
    test_set = dict()
    for file in training_filenames:
        training_set[file] = np.concatenate((haralick_training[file], hsv_training[file]))
    for file in test_filenames:
        test_set[file] = np.concatenate((haralick_test[file], hsv_test[file]))

    # Normalize the dataset with Standard Scaler
    ss = StandardScaler()
    ss.fit(list(training_set.values()))

    scaled_training = ss.transform(list(training_set.values()))
    for file, scaled in zip(training_filenames, scaled_training):
        training_set[file] = scaled

    scaled_test = ss.transform(list(test_set.values()))
    for file, scaled in zip(test_set.keys(), scaled_test):
        test_set[file] = scaled

    # Train the classifier
    print("Training the classifier...")
    if_clf = IsolationForest()
    if_clf.fit(list(training_set.values()))
    print("Classifier is ready!")

    # Begin prediction
    print("Labeling started.")
    for file in test_set.keys():
        prediction = if_clf.predict(test_set[file].reshape(1, -1))
        if prediction == 1:
            prediction = 'Biofilm'
        else:
            prediction = 'Other'

        image = cv2.imread(test_path + "/" + file)

        # show the label
        cv2.putText(image, prediction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)

        # display the output image
        cv2.imshow("Test_Image", image)
        cv2.waitKey(0)
