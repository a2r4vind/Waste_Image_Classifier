import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import os
import matplotlib.pyplot as plt
from collections import defaultdict

# experiment name
EXP_NAME = "exp4_resnet18_balanced_finetune_differential_lrs_weight_decay" # change for each experiment
print(f"Evaluating model for experiment {EXP_NAME}...")
print("=== Evaluation Started ===")

# Config
BATCH_SIZE = 32
NUM_CLASSES = 4

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Transforms
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ]
)

# directory for models 
MODEL_DIR = "/home/akki2404/CV_Project/Waste_Image_Classifier/models"
# Create the directory if it doesn't exist
os.makedirs(MODEL_DIR, exist_ok=True)
# path to saved model
model_path = os.path.join(MODEL_DIR, f"{EXP_NAME}.pth")

# Load the test data
test_data = datasets.ImageFolder("/home/akki2404/CV_Project/Waste_Image_Classifier/data/test", transform=transform)

# test DataLoader
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

# Load the model
model = models.resnet18(weights="IMAGENET1K_V1")
model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)

# checkpoint = torch.load(model_path, map_location=device) # from exp5 onwards, loading model using checkpoint dict to also get optimizer state and epoch info for better reproducibility and future fine-tuning
# model.load_state_dict(checkpoint["model_state_dict"])
# Earlier way of loading model without checkpoint dict did upto exp1 to exp4
model.load_state_dict(torch.load(f"/home/akki2404/CV_Project/Waste_Image_Classifier/models/{EXP_NAME}.pth", map_location=device))
model.to(device)

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
report = classification_report(all_labels, all_preds, target_names=test_data.classes)
print(report)

# Accuracy
accuracy = (np.array(all_preds) == np.array(all_labels)).mean()
print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

# function to show misclassified samples
def show_misclassified(misclassified, num=12):
    plt.figure(figsize=(12, 10))

    for i in range(min(num, len(misclassified))):
        img = misclassified[i]["image"]
        true = misclassified[i]["true"]
        pred = misclassified[i]["pred"]

        img = img.permute(1, 2, 0).numpy() # CxHxW to HxWxC

        # denormalize for proper display
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)

        plt.subplot(3, 4, i + 1)
        plt.imshow(img)
        plt.title(f"True: {test_data.classes[true]} | Pred: {test_data.classes[pred]}")
        plt.axis("off")

    plt.tight_layout()

    output_path = os.path.join(RESULTS_DIR, f"{EXP_NAME}_misclassified.png")
    plt.savefig(output_path)
    print(f"Misclassified images saved at:  {output_path}")


# Error Grouping 
error_groups = defaultdict(int)

for item in misclassified:
    key = (item["true"], item["pred"])
    error_groups[key] += 1

print("\nError Analysis (True -> Predicted):")
for (t,p), count in sorted(error_groups.items(), key=lambda x: x[1], reverse=True):
    print(f"{test_data.classes[t]} -> {test_data.classes[p]}: {count} samples")

# Save results to file
RESULTS_DIR = "/home/akki2404/CV_Project/Waste_Image_Classifier/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

with open(f"{RESULTS_DIR}/{EXP_NAME}_results.txt", "w") as f:
    f.write(f"Experiment: {EXP_NAME}\n\n")
    f.write("Confusion Matrix:\n")
    f.write(str(cm))
    f.write("\n\nClassification Report:\n")
    f.write(report)
    f.write(f"\n\nTest Accuracy: {accuracy * 100:.2f}%")
    f.write("\n\nError Analysis (True -> Predicted : Sample Count):\n")
    for (t,p), count in sorted(error_groups.items(), key=lambda x: x[1], reverse=True):
        f.write(f"{test_data.classes[t]} -> {test_data.classes[p]}: {count} samples\n")


# calling function to show misclassified samples
print(f"\nTotal Misclassified: {len(misclassified)}")
if len(misclassified) == 0:
    print("No misclassified samples to display.")
else:
    show_misclassified(misclassified, num=12)