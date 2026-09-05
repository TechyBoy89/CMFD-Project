# ==========================================
# prepare_dataset.py
# Extract and Save SIFT Features
# ==========================================

import os
import cv2
import numpy as np

from config import DATASET_PATH, IMAGE_SIZE
from sift_features import extract_sift_features

X = []
Y = []

print("Preparing Dataset...\n")

for file in sorted(os.listdir(DATASET_PATH)):

    if not file.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    # Skip Mask Images
    if "_B" in file or "_M" in file:
        continue

    path = os.path.join(DATASET_PATH, file)

    image = cv2.imread(path)

    if image is None:
        continue

    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))

    keypoints, descriptors = extract_sift_features(image)

    # If no descriptors found
    if descriptors is None:
        feature = np.zeros(128)

    else:
        feature = np.mean(descriptors, axis=0)

    X.append(feature)

    # Label
    if "_F" in file:
        Y.append(1)
    else:
        Y.append(0)

X = np.array(X)
Y = np.array(Y)

print("\nSaving Features...")

np.save("features/X.npy", X)
np.save("features/Y.npy", Y)

print("Done!")

print("Feature Shape :", X.shape)
print("Label Shape :", Y.shape)