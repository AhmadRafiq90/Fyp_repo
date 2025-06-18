from PIL import Image
from torchvision import transforms
import torch
import torch.nn as nn
from torchvision import models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image transform (same as training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Load the model
model = models.resnet18(pretrained=False)
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 1),
    nn.Sigmoid()
)
model.load_state_dict(torch.load('Binary_classifier_ResNet18.pth', map_location=device))
model = model.to(device)
model.eval()

def predict_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0).to(device)

    output = model(image)
    prob = output.item()

    prob = 1 - prob

    if prob >= 0.5:
        prediction = 'Leaf'
    else:
        prediction = 'Not Leaf'
    
    print(f'Prediction: {prediction} (Confidence: {prob:.4f})')

# Example usage
predict_image('No_leaf/test.jpeg')
