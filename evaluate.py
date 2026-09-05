# ==========================================
# evaluate.py
# Evaluate Hybrid Model
# ==========================================

import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

print("Loading Features...")

X = np.load("features/X.npy")
Y = np.load("features/Y.npy")

# Normalize
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split dataset (same random state as training)
X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.20,
    random_state=42,
    stratify=Y
)

# Load trained model
model = tf.keras.models.load_model("models/cmfd_model.keras")

# Predict
predictions = model.predict(X_test)

# Convert probabilities into labels
Y_pred = (predictions > 0.5).astype(int)

print("\n========== RESULTS ==========\n")

print("Accuracy  :", accuracy_score(Y_test, Y_pred))
print("Precision :", precision_score(Y_test, Y_pred))
print("Recall    :", recall_score(Y_test, Y_pred))
print("F1 Score  :", f1_score(Y_test, Y_pred))

print("\nConfusion Matrix\n")
print(confusion_matrix(Y_test, Y_pred))

print("\nClassification Report\n")
print(classification_report(Y_test, Y_pred))