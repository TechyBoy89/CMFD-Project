# ==========================================
# sift_features.py
# Extract SIFT Features
# ==========================================

import cv2
import numpy as np


def extract_sift_features(image):

    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Create SIFT detector
    sift = cv2.SIFT_create()

    # Detect keypoints and descriptors
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    # If no descriptors found
    if descriptors is None:
        descriptors = np.zeros((1, 128))

    return keypoints, descriptors


if __name__ == "__main__":

    image = cv2.imread("dataset/119_O.png")

    keypoints, descriptors = extract_sift_features(image)

    print("Number of Keypoints :", len(keypoints))
    print("Descriptor Shape :", descriptors.shape)