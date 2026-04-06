import shutil
import random
import numpy as np
import os

# fix randomness for reproducibility
random.seed(42)
np.random.seed(42)

# Paths
SOURCE_DIR = "/home/akki2404/CV_Project/waste_dataset" # path to the dataset directory
TARGET_DIR = "/home/akki2404/CV_Project/Waste_Image_Classifier/data" # path to the directory where the split data will be stored

# prevent re-splitting if data already splitted
if os.path.exists(f"{TARGET_DIR}/train") and os.path.exists(f"{TARGET_DIR}/val"):
    print("Data already split. Skipping splitting process.")
    exit()



# Classes 
CLASSES = ["glass", "plastic", "metal", "paper"]

# Split Ratio
SPLIT_RATIO = 0.8

for cls in CLASSES:
    os.makedirs(f"{TARGET_DIR}/train/{cls}", exist_ok=True)
    os.makedirs(f"{TARGET_DIR}/val/{cls}", exist_ok=True)

    images = os.listdir(f"{SOURCE_DIR}/{cls}")
    random.shuffle(images)

    split_index = int(len(images) * SPLIT_RATIO)

    train_images = images[:split_index]
    val_images = images[split_index:]

    for img in train_images:
        src = f"{SOURCE_DIR}/{cls}/{img}"
        dst = f"{TARGET_DIR}/train/{cls}/{img}"
        shutil.copy(src, dst)

    for img in val_images:
        src = f"{SOURCE_DIR}/{cls}/{img}"
        dst = f"{TARGET_DIR}/val/{cls}/{img}"
        shutil.copy(src, dst)

    print(f"Class '{cls}': {len(train_images)} images for training, {len(val_images)} images for validation.")

print("Data splitting completed successfully!")

