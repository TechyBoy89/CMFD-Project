import tkinter as tk
from tkinter import filedialog, messagebox
import os
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageTk

from sift_features import extract_sift_features
from localization import find_copy_move
from config import IMAGE_SIZE, MODEL_PATH


# ==========================================
# Load Model
# ==========================================

print("Loading Model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model Loaded Successfully!")


# ==========================================
# Main Window
# ==========================================

window = tk.Tk()

window.title("Copy-Move Forgery Detection")
window.geometry("600x850")


# ==========================================
# Global Localization Image
# ==========================================

localized_image = None


# ==========================================
# Select Image Function
# ==========================================

def select_image():

    global localized_image

    image_path = filedialog.askopenfilename(
        title="Select Image",
        filetypes=[
            ("Image Files", "*.png *.jpg *.jpeg")
        ]
    )

    if image_path == "":
        return

    filename = os.path.basename(image_path)

    selected_label.config(
        text="Selected Image : " + filename
    )


    # ======================================
    # Read Image
    # ======================================

    image = cv2.imread(image_path)

    if image is None:

        messagebox.showerror(
            "Error",
            "Cannot read selected image."
        )

        return


    image = cv2.resize(
        image,
        (IMAGE_SIZE, IMAGE_SIZE)
    )


    # ======================================
    # Extract SIFT Features
    # ======================================

    keypoints, descriptors = extract_sift_features(image)

    if descriptors is None:

        feature = np.zeros(128)

    else:

        feature = np.mean(
            descriptors,
            axis=0
        )


    # ======================================
    # Normalize Feature
    # ======================================

    scaler_mean = np.load(
        "features/scaler_mean.npy"
    )

    scaler_scale = np.load(
        "features/scaler_scale.npy"
    )

    feature = (
        feature - scaler_mean
    ) / scaler_scale

    feature = feature.reshape(
        1,
        128
    )


    # ======================================
    # Prediction
    # ======================================

    prediction = model.predict(
        feature,
        verbose=0
    )

    score = prediction[0][0]


    # ======================================
    # Result
    # ======================================

    if score >= 0.5:

        result = "FORGED IMAGE"

        confidence = score * 100

    else:

        result = "ORIGINAL IMAGE"

        confidence = (1 - score) * 100


    # ======================================
    # Display Prediction
    # ======================================

    result_label.config(
        text="Prediction : " + result
    )

    confidence_label.config(
        text="Confidence : {:.2f}%".format(
            confidence
        )
    )


    # ======================================
    # SIFT Localization
    # ======================================

    localized_image, detected = find_copy_move(
        image
    )


    # ======================================
    # Localization Result
    # ======================================

    if result == "FORGED IMAGE" and detected:

        localization_label.config(
            text="Localization : DETECTED"
        )

    else:

        localization_label.config(
            text="Localization : NOT DETECTED"
        )


    # ======================================
    # Display Localization Image
    # ======================================

    display_image = cv2.cvtColor(
        localized_image,
        cv2.COLOR_BGR2RGB
    )

    display_image = Image.fromarray(
        display_image
    )

    display_image = display_image.resize(
        (300, 300)
    )

    photo = ImageTk.PhotoImage(
        display_image
    )

    image_label.config(
        image=photo
    )

    image_label.image = photo


# ==========================================
# Clear / Reset Function
# ==========================================

def clear_result():

    global localized_image

    localized_image = None

    selected_label.config(
        text="No Image Selected"
    )

    result_label.config(
        text="Prediction : ---"
    )

    confidence_label.config(
        text="Confidence : ---"
    )

    localization_label.config(
        text="Localization : ---"
    )

    image_label.config(
        image=""
    )

    image_label.image = None


# ==========================================
# Save Result Function
# ==========================================

def save_result():

    if localized_image is None:

        messagebox.showwarning(
            "No Result",
            "Please select and analyze an image first."
        )

        return


    save_path = filedialog.asksaveasfilename(
        title="Save Localization Result",
        defaultextension=".png",
        filetypes=[
            ("PNG Image", "*.png"),
            ("JPG Image", "*.jpg")
        ]
    )

    if save_path == "":
        return


    # ======================================
    # Save Localization Image
    # ======================================

    cv2.imwrite(
        save_path,
        localized_image
    )


    messagebox.showinfo(
        "Saved",
        "Result saved successfully!"
    )


# ==========================================
# Title
# ==========================================

title = tk.Label(
    window,
    text="COPY-MOVE FORGERY DETECTION",
    font=("Arial", 18, "bold")
)

title.pack(pady=20)


# ==========================================
# Select Button
# ==========================================

select_button = tk.Button(
    window,
    text="SELECT IMAGE",
    font=("Arial", 12, "bold"),
    command=select_image,
    width=20,
    height=2
)

select_button.pack(pady=10)


# ==========================================
# Selected Image
# ==========================================

selected_label = tk.Label(
    window,
    text="No Image Selected",
    font=("Arial", 12)
)

selected_label.pack(pady=10)


# ==========================================
# Prediction
# ==========================================

result_label = tk.Label(
    window,
    text="Prediction : ---",
    font=("Arial", 14, "bold")
)

result_label.pack(pady=10)


# ==========================================
# Confidence
# ==========================================

confidence_label = tk.Label(
    window,
    text="Confidence : ---",
    font=("Arial", 13)
)

confidence_label.pack(pady=10)


# ==========================================
# Localization
# ==========================================

localization_label = tk.Label(
    window,
    text="Localization : ---",
    font=("Arial", 13)
)

localization_label.pack(pady=10)


# ==========================================
# Image Display
# ==========================================

image_label = tk.Label(
    window
)

image_label.pack(pady=10)


# ==========================================
# Clear Button
# ==========================================

clear_button = tk.Button(
    window,
    text="CLEAR",
    font=("Arial", 12, "bold"),
    command=clear_result,
    width=20,
    height=2
)

clear_button.pack(pady=10)


# ==========================================
# Save Result Button
# ==========================================

save_button = tk.Button(
    window,
    text="SAVE RESULT",
    font=("Arial", 12, "bold"),
    command=save_result,
    width=20,
    height=2
)

save_button.pack(pady=10)


# ==========================================
# Start GUI
# ==========================================

window.mainloop()