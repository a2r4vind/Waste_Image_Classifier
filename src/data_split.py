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
if (
    os.path.exists(f"{TARGET_DIR}/train") and 
    os.path.exists(f"{TARGET_DIR}/val") and 
    os.path.exists(f"{TARGET_DIR}/test")
):
    print("Data already split. Skipping splitting process.")
    exit()

# Classes 
CLASSES = ["glass", "plastic", "metal", "paper"]

# Split Ratio
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

for cls in CLASSES:
    os.makedirs(f"{TARGET_DIR}/train/{cls}", exist_ok=True)
    os.makedirs(f"{TARGET_DIR}/val/{cls}", exist_ok=True)
    os.makedirs(f"{TARGET_DIR}/test/{cls}", exist_ok=True)

    images = os.listdir(f"{SOURCE_DIR}/{cls}")
    random.shuffle(images)

    total = len(images)

    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    train_images = images[:train_end]
    val_images = images[train_end:val_end]
    test_images = images[val_end:]

    # copy files
    for img in train_images:
        src = f"{SOURCE_DIR}/{cls}/{img}"
        dst = f"{TARGET_DIR}/train/{cls}/{img}"
        shutil.copy(src, dst)

    for img in val_images:
        src = f"{SOURCE_DIR}/{cls}/{img}"
        dst = f"{TARGET_DIR}/val/{cls}/{img}"
        shutil.copy(src, dst)

    for img in test_images:
        src = f"{SOURCE_DIR}/{cls}/{img}"
        dst = f"{TARGET_DIR}/test/{cls}/{img}"
        shutil.copy(src, dst)

    print(f"Class '{cls}': {len(train_images)} images for training, {len(val_images)} images for validation, {len(test_images)} images for testing.")

print("Data splitting completed successfully!")

