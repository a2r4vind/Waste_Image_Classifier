import torch
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import os
from collections import defaultdict
from data import get_test_loader
from utils import get_device, load_model, get_model_path, show_misclassified
from model import get_model
import yaml
import argparse

# load config
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True, help="path to the config file")
args = parser.parse_args()

with open(args.config, "r") as f:
    config = yaml.safe_load(f)

# Configs
DATA_DIR = config["data"]["data_dir"] # Data directory
EXP_NAME = config["experiment"]["name"] # experiment name change for each experiment
BATCH_SIZE = int(config["data"]["batch_size"])
MODEL_NAME = config["model"]["name"]
RESULTS_DIR = config["evaluation"]["results_dir"] # results directory
os.makedirs(RESULTS_DIR, exist_ok=True) # create results directory if it doesn't exist

print(f"Evaluating model for experiment {EXP_NAME}...")
print("=== Evaluation Started ===")

# Device
device = get_device()

# path to saved model
model_path = get_model_path(EXP_NAME)

# test DataLoader, NUM_CLASSES and CLASS_NAMES
test_loader, NUM_CLASSES, CLASS_NAMES = get_test_loader(DATA_DIR, BATCH_SIZE)

# get the model for evaluation
model = get_model(MODEL_NAME, NUM_CLASSES, freeze_backbone=False) # freeze_backbone=False to avoid freezing layers

# load the saved model weights
model = load_model(model, model_path, device)
model.to(device)
model.eval()

# for storing misclassified samples
misclassified = []

# Evaluation
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        # Store misclassified samples
        for i in range(len(preds)):
            if preds[i] != labels[i]:
                misclassified.append({
                    "image": images[i].cpu(),
                    "true" : labels[i].cpu().item(),
                    "pred" : preds[i].cpu().item()
                })


# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
print("\nConfusion Matrix:")
print(cm) 

# Classification Report 
print("\nClassification Report:")
report = classification_report(all_labels, all_preds, target_names=CLASS_NAMES)
print(report)

# Accuracy
accuracy = (np.array(all_preds) == np.array(all_labels)).mean()
print(f"\nTest Accuracy: {accuracy * 100:.2f}%")


# Error Grouping 
error_groups = defaultdict(int)

for item in misclassified:
    key = (item["true"], item["pred"])
    error_groups[key] += 1

print("\nError Analysis (True -> Predicted):")
for (t,p), count in sorted(error_groups.items(), key=lambda x: x[1], reverse=True):
    print(f"{CLASS_NAMES[t]} -> {CLASS_NAMES[p]}: {count} samples")

# Save results to file
with open(f"{RESULTS_DIR}/{EXP_NAME}_results.txt", "w") as f:
    f.write(f"Experiment: {EXP_NAME}\n\n")
    f.write("Confusion Matrix:\n")
    f.write(str(cm))
    f.write("\n\nClassification Report:\n")
    f.write(report)
    f.write(f"\n\nTest Accuracy: {accuracy * 100:.2f}%")
    f.write("\n\nError Analysis (True -> Predicted : Sample Count):\n")
    for (t,p), count in sorted(error_groups.items(), key=lambda x: x[1], reverse=True):
        f.write(f"{CLASS_NAMES[t]} -> {CLASS_NAMES[p]}: {count} samples\n")
    


# calling show_misclassified() function to show misclassified samples
print(f"\nTotal Misclassified: {len(misclassified)}")
if len(misclassified) == 0:
    print("No misclassified samples to display.")
else:
    show_misclassified(misclassified, RESULTS_DIR, EXP_NAME, CLASS_NAMES, num=12)