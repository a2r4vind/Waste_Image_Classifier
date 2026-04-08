import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import os

# experiment name
EXP_NAME = "exp2_resnet18_aug_layer4_finetune" # change for each experiment
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

# Load the test data
test_data = datasets.ImageFolder("/home/akki2404/CV_Project/Waste_Image_Classifier/data/test", transform=transform)

# test DataLoader
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

# Load the model
model = models.resnet18(weights="IMAGENET1K_V1")
model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)
model.load_state_dict(torch.load(f"/home/akki2404/CV_Project/Waste_Image_Classifier/models/{EXP_NAME}.pth", map_location=device))
model.to(device)

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