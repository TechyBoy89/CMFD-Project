import os
import cv2
import numpy as np
import tensorflow as tf

from sift_features import extract_sift_features
from config import IMAGE_SIZE, MODEL_PATH


# --------------------------------
# Load model
# --------------------------------

print("Loading Model...")

model = tf.keras.models.load_model(MODEL_PATH)


# --------------------------------
# Load scaler
# --------------------------------

scaler_mean = np.load(
    "features/scaler_mean.npy"
)

scaler_scale = np.load(
    "features/scaler_scale.npy"
)


# --------------------------------
# Dataset folder
# --------------------------------

DATASET_FOLDER = r"C:\Users\krish\Downloads\dataset"


# --------------------------------
# Counters
# --------------------------------

original_correct = 0
original_wrong = 0

forged_correct = 0
forged_wrong = 0


# --------------------------------
# Test images
# --------------------------------

files = os.listdir(DATASET_FOLDER)

for filename in files:

    # Only test original and forged
    # base images

    if not (
        filename.endswith("_O.png")
        or filename.endswith("_F.png")
    ):
        continue

    image_path = os.path.join(
        DATASET_FOLDER,
        filename
    )

    image = cv2.imread(image_path)

    if image is None:
        continue

    image = cv2.resize(
        image,
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    # --------------------------------
    # SIFT
    # --------------------------------

    keypoints, descriptors = extract_sift_features(
        image
    )

    if descriptors is None:

        feature = np.zeros(128)

    else:

        feature = np.mean(
            descriptors,
            axis=0
        )

    # --------------------------------
    # Normalize
    # --------------------------------

    feature = (
        feature - scaler_mean
    ) / scaler_scale

    feature = feature.reshape(
        1,
        128
    )

    # --------------------------------
    # Prediction
    # --------------------------------

    prediction = model.predict(
        feature,
        verbose=0
    )

    score = prediction[0][0]

    if score >= 0.5:

        predicted = 1

    else:

        predicted = 0

    # --------------------------------
    # Actual label
    # --------------------------------

    if "_O.png" in filename:

        actual = 0

    else:

        actual = 1

    # --------------------------------
    # Check result
    # --------------------------------

    if actual == 0:

        if predicted == 0:

            original_correct += 1

        else:

            original_wrong += 1

    else:

        if predicted == 1:

            forged_correct += 1

        else:

            forged_wrong += 1


# --------------------------------
# Results
# --------------------------------

print()
print("==============================")
print("       TEST RESULTS")
print("==============================")

print()

print(
    "Original Correct :",
    original_correct
)

print(
    "Original Wrong   :",
    original_wrong
)

print(
    "Forged Correct   :",
    forged_correct
)

print(
    "Forged Wrong     :",
    forged_wrong
)

print()