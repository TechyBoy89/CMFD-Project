# ==========================================
# dataset_loader.py
# Scan CoMoFoD Dataset
# ==========================================

import os
from config import DATASET_PATH

# Counters
total_images = 0
original_images = 0
forged_images = 0
mask_images = 0

# Store information
dataset = []

# Check if dataset folder exists
if not os.path.exists(DATASET_PATH):
    print("Dataset folder not found!")
    print("Expected path:", DATASET_PATH)
    exit()

# Read every file
for file in sorted(os.listdir(DATASET_PATH)):

    # Only image files
    if not file.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    total_images += 1

    # Skip masks for training
    if "_B" in file or "_M" in file:
        mask_images += 1
        continue

    # Original Image
    if "_O" in file:
        label = 0
        original_images += 1

    # Forged Image
    elif "_F" in file:
        label = 1
        forged_images += 1

    else:
        continue

    dataset.append({
        "filename": file,
        "label": label
    })

print("=" * 50)
print("CoMoFoD Dataset Summary")
print("=" * 50)

print("Total Image Files :", total_images)
print("Original Images  :", original_images)
print("Forged Images    :", forged_images)
print("Mask Images      :", mask_images)

print("\nFirst 10 Images\n")

for item in dataset[:10]:
    print(item)

print("\nDataset Ready.")