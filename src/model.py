from torchvision import models
from torch import nn


def get_model(model_name, num_classes, freeze_backbone=True):
    if model_name == "resnet18":
        model = models.resnet18(weights="IMAGENET1K_V1")
    elif model_name == "resnet34":
        model = models.resnet34(weights="IMAGENET1K_V1")
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
    
    if freeze_backbone:
        # Freeze all layers
        for param in model.parameters():
            param.requires_grad = False

        # fine-tuning -> modified layer 4 
        for param in model.layer4.parameters():  # in exp2, exp4
            param.requires_grad = True

    # Replace the final layer
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model
