import os
import torch
import matplotlib.pyplot as plt
import numpy as np

# Utility functions for model saving
def save_model(model, num_classes, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "num_classes": num_classes,
    }, path)

# Utility function to get device
def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device 

# Utility function to load model 
def load_model(model, path, device):
    checkpoint = torch.load(path, map_location=device)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        # for older experiments (exp1–exp4)
        model.load_state_dict(checkpoint)

    return model

# Utility function to get model path
def get_model_path(exp_name):
    # directory for models
    model_dir = "/home/akki2404/CV_Project/Waste_Image_Classifier/models"
    # Create the directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{exp_name}.pth")
    return model_path

# Utility function to show misclassified samples
def show_misclassified(misclassified, result_dir, exp_name, class_names, num=12):
    # create results directory if it doesn't exist
    os.makedirs(result_dir, exist_ok=True)
    plt.figure(figsize=(12, 10))

    for i in range(min(num, len(misclassified))):
        img = misclassified[i]["image"]
        true = misclassified[i]["true"]
        pred = misclassified[i]["pred"]

        img = img.cpu().permute(1, 2, 0).numpy() # CxHxW to HxWxC

        # denormalize for proper display
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)

        plt.subplot(3, 4, i + 1)
        plt.imshow(img)
        plt.title(f"True: {class_names[true]} | Pred: {class_names[pred]}")
        plt.axis("off")

    plt.tight_layout()

    output_path = os.path.join(result_dir, f"{exp_name}_misclassified.png")
    plt.savefig(output_path)
    plt.close()
    print(f"Misclassified images saved at:  {output_path}")