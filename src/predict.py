import torch
from torchvision import transforms
from PIL import Image
from model import get_model
from utils import load_model

# same normalization as validation transforms
def get_inference_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

# Funtion to predict the class of a single image
def predict_image(image, model, device, class_names):
    """
    image: PIL Image or path to image
    model: trained model
    device: cpu/cuda
    class_names: list of class labels
    """

    # load image
    if isinstance(image, str):
        image = Image.open(image).convert("RGB") # if input is path, load and convert to RGB
    else:
        image = image.convert("RGB")

    # get inference transform
    transform = get_inference_transform()

    # apply transform
    image = transform(image).unsqueeze(0).to(device)  # shape: (1, C, H, W)

    model.eval()
    with torch.no_grad():
        outputs = model(image)

        probs = torch.softmax(outputs, dim=1)
        k = min(3, probs.shape[1])
        top_probs, top_preds = torch.topk(probs, k=k)
        # confidence, pred = torch.max(probs, dim=1)

    # predicted_class = class_names[pred.item()]
    # confidence_score = confidence.item()

    # return predicted_class, confidence_score

    results = []
    for i in range(top_probs.size(1)):
        results.append({
            "class": class_names[top_preds[0][i].item()],
            "confidence": round(top_probs[0][i].item(), 4)
        })
    
    top_class = results[0]["class"]
    top_conf = results[0]["confidence"]

    return top_class, top_conf, results


# Function to load model for inference
def load_model_for_inference(model_name, model_path, num_classes, device):
    model = get_model(model_name, num_classes, freeze_backbone=False)
    model = load_model(model, model_path, device)
    model.to(device)
    model.eval()
    return model