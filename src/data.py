from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

def get_transforms(train=True):
    if train:
        return transforms.Compose(
            [
                # transforms.Resize((224, 224)), # not in exp2, exp4
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)), # in exp2, exp4
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(20), # 10 -> 20 (exp2, exp4) -> 10
                transforms.ColorJitter(
                    brightness=0.3,
                    contrast=0.3,
                    saturation=0.3
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                 mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ]
        )
    
def get_dataloaders(data_dir, batch_size=32):
    train_data = datasets.ImageFolder(
        os.path.join(data_dir, "train"),
        transform = get_transforms(train=True)
    )
    val_data = datasets.ImageFolder(
        os.path.join(data_dir, "val"),
        transform = get_transforms(train=False)
    )

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, len(train_data.classes), train_data.classes

def get_test_loader(data_dir, batch_size=32):
    test_data = datasets.ImageFolder(
        os.path.join(data_dir, "test"),
        transform=get_transforms(train=False)
    )
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return test_loader, len(test_data.classes), test_data.classes