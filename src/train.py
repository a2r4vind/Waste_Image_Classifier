import torch
from torchvision import datasets, transforms, models
from torch import nn, optim
from torch.utils.data import DataLoader
import os


# Transforms
train_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)), 
        # transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10), # 10 -> 20
        # transforms.ColorJitter(
        #     brightness=0.3,
        #     contrast=0.3,
        #     saturation=0.3
        # ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ]
)

val_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ]
)

# Data directory
DATA_DIR = "/home/akki2404/CV_Project/Waste_Image_Classifier/data"

# Dataset
train_data = datasets.ImageFolder(os.path.join(DATA_DIR,"train"), transform=train_transform)
val_data = datasets.ImageFolder(os.path.join(DATA_DIR,"val"), transform=val_transform)

# Config
BATCH_SIZE = 32
EPOCHS = 5 # 5 -> 10 
NUM_CLASSES = len(train_data.classes)

# DataLoader
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

# Model 
model = models.resnet18(weights="IMAGENET1K_V1") # resnet18 -> resnet34

# Freeze layers
for param in model.parameters():
    param.requires_grad = False

# fine-tuning -> modified layer 4 
# for param in model.layer4.parameters():
#     param.requires_grad = True

# Replace the final layer
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

#  train FC layer
for param in model.fc.parameters():
    param.requires_grad = True

# # train layer4
# for param in model.layer4.parameters():
#     param.requires_grad = True


# check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model.to(device)

# Loss 
criterion = nn.CrossEntropyLoss()

# Optimizer (only FC layer)
# Optimizer
# optimizer = optim.Adam(model.fc.parameters(), lr=0.001) # 0.001 => Higher LR helps faster convergence
# optimizer = optim.Adam(
#     list(model.fc.parameters()) + list(model.layer4.parameters()),
#     lr=0.001
# )
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# initialize best accuracy
best_acc = 0.0

# Directory to save the best model
MODEL_DIR = "/home/akki2404/CV_Project/Waste_Image_Classifier/models"
# Create the directory if it doesn't exist
os.makedirs(MODEL_DIR, exist_ok=True)
EXP_NAME = "exp1_resnet18_baseline_fc_only"
model_path = os.path.join(MODEL_DIR, f"{EXP_NAME}.pth")

print(f"Training ResNet18 for experiment {EXP_NAME}...")

# Training Loop
for epoch in range(EPOCHS):
    # Training phase
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    train_acc = 100 * correct / total 


    # Validation phase
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)

    val_acc = 100 * val_correct / val_total

    # Save the best model
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), model_path)
        print(f"Best model saved with accuracy: {best_acc:.2f}%")

    print(f"Epoch {epoch+1}")
    print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%")
    print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.2f}%")
    print("-" * 30)

print("Training completed successfully!")
