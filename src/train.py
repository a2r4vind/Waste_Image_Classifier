from data import get_dataloaders
from model import get_model
from engine import train_one_epoch, validate
from utils import save_model, get_device, get_model_path
from torch import nn, optim
import yaml
import argparse

# Load config
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True, help="Path to the config file")
args = parser.parse_args()

with open(args.config, "r") as f:
    config = yaml.safe_load(f)

# Configs
DATA_DIR = config["data"]["data_dir"] #  Data directory
BATCH_SIZE = int(config["data"]["batch_size"])
EPOCHS = int(config["training"]["epochs"]) # 5 (baseline model) -> 10 
MODEL_NAME = config["model"]["name"] # resnet18 -> resnet34
EXP_NAME = config["experiment"]["name"] # change for each experiment



# get device
device = get_device()

# get loaders and number of classes
train_loader, val_loader, NUM_CLASSES, CLASS_NAMES = get_dataloaders(DATA_DIR, BATCH_SIZE)

print(f"\nClasses: {CLASS_NAMES}")
print(f"Epochs: {EPOCHS}, Batch Size: {BATCH_SIZE}")

# get model
model = get_model(MODEL_NAME, NUM_CLASSES)

# send model to device
model.to(device)

# Loss 
criterion = nn.CrossEntropyLoss()

# Optimizer
# optimizer = optim.Adam(
#     list(model.fc.parameters()) + list(model.layer4.parameters()), # in exp2
#     lr=0.001
# )
# optimizer = optim.Adam(model.fc.parameters(), lr=0.0003) # lr = 0.001 (exp1) -> 0.0003 (exp3)
optimizer = optim.Adam(
    [
        {"params": model.fc.parameters(), "lr": float(config["optimizer"]["fc_lr"])}, # faster learning in exp4, exp5
        {"params": model.layer4.parameters(), "lr": float(config["optimizer"]["backbone_lr"])} # slower learning in exp4, exp5
    ],
    weight_decay=float(config["optimizer"]["weight_decay"]) # help reduce overfitting in exp4, exp5
)

# initialize best accuracy
best_acc = 0.0

# Path to save the best model
model_path = get_model_path(EXP_NAME)

print(f"\nTraining {MODEL_NAME} for experiment {EXP_NAME}...") # resnet18 -> resnet34
print("=== Training Started ===\n")

# Training Loop
for epoch in range(EPOCHS):
    # Training phase
    train_avg_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)

    # Validation phase
    val_avg_loss, val_acc = validate(model, val_loader, criterion, device)

    # Save the best model
    if val_acc > best_acc:
        best_acc = val_acc
        save_model(model, NUM_CLASSES, model_path)
        print(f"Best model saved with accuracy: {best_acc:.2f}%")

    print(f"Epoch {epoch+1}")
    print(f"Train Loss: {train_avg_loss:.4f}, Train Accuracy: {train_acc:.2f}%")
    print(f"Val Loss: {val_avg_loss:.4f}, Val Accuracy: {val_acc:.2f}%")
    print("-" * 30)

print("Training completed successfully!")
