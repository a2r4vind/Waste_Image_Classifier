from predict import predict_image, load_model_for_inference
from utils import get_device, get_model_path

device = get_device()

model = load_model_for_inference(
    model_name="resnet34",
    model_path=get_model_path("exp6_resnet34_config"),
    num_classes=4,
    device=device
)

class_names = ["glass", "metal", "paper", "plastic"]

try:
    top_class, top_conf, results = predict_image(
        "/home/akki2404/CV_Project/Waste_Image_Classifier/test_sample.jpg", # test.jpg is sample image for testing
        model,
        device,
        class_names
    )

    print(f"\nTop Prediction: {top_class} ({top_conf*100:.2f}%)\n")

    print("Top Predictions:")
    for r in results:
        print(f"{r['class']}: {r['confidence']*100:.2f}%")

except Exception as e:
    print(f"Error during prediction: {e}")


