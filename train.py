import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import *
from model import build_model

print("Loading Features...")

X = np.load("features/X.npy")
Y = np.load("features/Y.npy")

print("Dataset Loaded")

print(X.shape)
print(Y.shape)
# Normalize

scaler = StandardScaler()

X = scaler.fit_transform(X)

# Save scaler values for prediction
np.save("features/scaler_mean.npy", scaler.mean_)
np.save("features/scaler_scale.npy", scaler.scale_)

# Save scaler values for prediction
np.save("features/scaler_mean.npy", scaler.mean_)
np.save("features/scaler_scale.npy", scaler.scale_)

print("Scaler Saved")

# Split

X_train, X_test, Y_train, Y_test = train_test_split(

    X,
    Y,

    test_size=0.20,

    random_state=42,

    stratify=Y

)

model = build_model()

history = model.fit(

    X_train,

    Y_train,

    validation_data=(X_test, Y_test),

    epochs=EPOCHS,

    batch_size=BATCH_SIZE

)

loss, accuracy = model.evaluate(X_test, Y_test)

print()

print("Accuracy =", accuracy)

model.save(MODEL_PATH)

# ==========================================
# Plot Accuracy
# ==========================================

plt.figure(figsize=(8,5))

plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")

plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.legend()

plt.grid(True)

plt.savefig("outputs/accuracy_graph.png")

plt.close()

# ==========================================
# Plot Loss
# ==========================================

plt.figure(figsize=(8,5))

plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")

plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend()

plt.grid(True)

plt.savefig("outputs/loss_graph.png")

plt.close()

print("\nGraphs Saved Successfully!")

print("Model Saved")