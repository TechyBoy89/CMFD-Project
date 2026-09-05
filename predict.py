# ==========================================
# predict.py
# Select Image and Predict
# ==========================================

import os
import cv2
import numpy as np
import tensorflow as tf

from tkinter import Tk
from tkinter.filedialog import askopenfilename

from sift_features import extract_sift_features
from localization import find_copy_move
from config import IMAGE_SIZE, MODEL_PATH

# ----------------------------
# Load Model
# ----------------------------
print("Loading Model...")
model = tf.keras.models.load_model(MODEL_PATH)

# ----------------------------
# Select Image
# ----------------------------
Tk().withdraw()

image_path = askopenfilename(
    title="Select an Image",
    filetypes=[("Image Files", "*.png *.jpg *.jpeg")]
)

if image_path == "":
    print("No image selected.")
    exit()

filename = os.path.basename(image_path)

print("\nSelected Image :", filename)

# ----------------------------
# Read Image
# ----------------------------
image = cv2.imread(image_path)

if image is None:
    print("Cannot read image.")
    exit()

image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))

# ----------------------------
# Extract SIFT Features
# ----------------------------
keypoints, descriptors = extract_sift_features(image)

if descriptors is None:
    feature = np.zeros(128)
else:
    feature = np.mean(descriptors, axis=0)

# --------------------------------
# Normalize using training scaler
# --------------------------------

scaler_mean = np.load("features/scaler_mean.npy")
scaler_scale = np.load("features/scaler_scale.npy")

feature = (feature - scaler_mean) / scaler_scale

feature = feature.reshape(1, 128)

# ----------------------------
# Prediction
# ----------------------------
prediction = model.predict(feature)

score = prediction[0][0]

if score >= 0.5:

    result = "FORGED IMAGE"
    confidence = score * 100

else:

    result = "ORIGINAL IMAGE"
    confidence = (1 - score) * 100

print("\n================================")
print("       CMFD DETECTION RESULT")
print("================================")

print("Selected Image :", filename)
print("Prediction     :", result)
print("Confidence     : {:.2f}%".format(confidence))

# ----------------------------
# Forgery Localization
# ----------------------------

print("\nRunning SIFT localization...")

localized_image = find_copy_move(image)

print("================================")


# ----------------------------
# Put Prediction Text
# ----------------------------

display = image.copy()

cv2.putText(
    display,
    result,
    (10, 30),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.8,
    (0, 255, 0),
    2
)

cv2.putText(
    display,
    "Confidence: {:.2f}%".format(confidence),
    (10, 60),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.6,
    (0, 255, 0),
    2
)

# ----------------------------
# Combine Image + Mask
# ----------------------------

combined = np.hstack((display, localized_image))

cv2.imshow("Prediction (Left)   |   Localization (Right)", combined)

cv2.waitKey(0)
cv2.destroyAllWindows()