import torch
from torchvision import datasets, transforms, models
from torch import nn, optim
from torch.utils.data import DataLoader

# Config
BATCH_SIZE = 32
EPOCHS = 5
NUM_CLASSES = 4

# Transforms
train_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
    ]
)

val_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)

# Dataset
train_data = datasets.ImageFolder("/home/akki2404/CV_Project/Waste_Image_Classifier/data/train", transform=train_transform)
val_data = datasets.ImageFolder("/home/akki2404/CV_Project/Waste_Image_Classifier/data/val", transform=val_transform)

# DataLoader
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

# Model 
model = models.resnet18(weights="IMAGENET1K_V1")

# Freeze layers
for param in model.parameters():
    param.requires_grad = False

# Replace the final layer
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

# check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Loss 
criterion = nn.CrossEntropyLoss()

# Optimizer 
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

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

    print(f"Epoch {epoch+1}")
    print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%")
    print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.2f}%")
    print("-" * 30)

print("Training completed successfully!")


# Save the model
torch.save(model.state_dict(), "/home/akki2404/CV_Project/Waste_Image_Classifier/models/resnet18_model.pth")
