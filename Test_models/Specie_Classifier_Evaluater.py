import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import argparse

# List of species (classes)
classes = [
    "Ailanthus altissima Mill Swingle(182)",
    "Aloe Vera",
    "Alstonia Scholaris",
    "Apple",
    "Arjun",
    "Blueberry",
    "Buxus sempervirens L(200)",
    "Cherry",
    "Corn",
    "Corylus avellana L(199)",
    "Cotinus coggygria Scop(200)",
    "Crataegus monogyna Jacq(200)",
    "Fraxinus angustifolia Vahi(200)",
    "Grape",
    "Guava",
    "Hedera helix L(200)",
    "Jamun",
    "Jatropha",
    "Kale",
    "Laurus nobilis L(200)",
    "Lemon",
    "Mango",
    "Orange",
    "Peach",
    "Pepper Bell",
    "Phillyrea angustifolia L(200)",
    "Pistacia lentiscus L(200)",
    "Pittosporum tobira Thumb WTAiton(200)",
    "Pomegranate",
    "Pongamia Pinnata",
    "Populus alba L(200)",
    "Populus nigra L(200)",
    "Potato",
    "Quercus ilex L(200)",
    "Raspberry",
    "Ruscus aculeatus L(200)",
    "Soybean",
    "Strawberry",
    "Tomato"
]

# Function to configure and load binary classifier
def load_binary_classifier(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 1),
        nn.Sigmoid()
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

# Function to configure and load plant classifier
def load_plant_classifier(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet34(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

# Image preprocessing
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet mean
            std=[0.229, 0.224, 0.225]    # ImageNet std
        )
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)

# Prediction function
def predict(image_path, model):
    image_tensor = preprocess_image(image_path)
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted_idx = torch.max(probabilities, 0)
        predicted_class = classes[predicted_idx.item()]
        return predicted_class, confidence.item()

# CLI interface
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plant Species Prediction")
    parser.add_argument("image_path", type=str, help="Path to input image")
    args = parser.parse_args()

    # Load binary plant/non-plant classifier
    binary_classifier_model_path = "Binary_classifier_ResNet18.pth"
    binary_classifier_model = load_binary_classifier(binary_classifier_model_path)

    # Load plant species classifier
    plant_classifier_model_path = "resnet34_plant_classifier.pth"
    plant_classifier_model = load_plant_classifier(plant_classifier_model_path)

    # Preprocess image once
    image_tensor = preprocess_image(args.image_path)

    # Step 1: Binary classification
    with torch.no_grad():
        binary_output = binary_classifier_model(image_tensor)
        binary_prob = binary_output.item()  # Already sigmoid applied in model

    # Step 2: Decide and predict species if plant
    binary_prob = 1 - binary_prob

    if binary_prob >= 0.5:
        predicted_class, confidence = predict(args.image_path, plant_classifier_model)
        print(f"✅ It's a **plant** with probability {binary_prob:.4f}")
        print(f"🌿 Predicted Species: {predicted_class}")
        print(f"📊 Confidence Score: {confidence:.4f}")
    else:
        print(f"❌ Not a plant (confidence: {1 - binary_prob:.4f})")
