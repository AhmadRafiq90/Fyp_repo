import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define transforms for training and validation
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Create dataloaders from a single folder with 80/20 split
def create_dataloaders(data_dir, batch_size=32, train_ratio=0.8):
    """Create training and validation dataloaders from a single data directory using 80/20 split"""
    # Create a dataset with NO transforms initially
    dataset = ImageFolder(data_dir, transform=None)
    
    # Calculate train/val split sizes
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    
    # Create a fixed random generator for reproducibility
    generator = torch.Generator().manual_seed(42)
    
    # Split dataset
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
    
    # Create custom datasets with appropriate transforms
    class TransformDataset(Dataset):
        def __init__(self, dataset, transform=None):
            self.dataset = dataset
            self.transform = transform
            
        def __getitem__(self, index):
            image, label = self.dataset[index]
            if self.transform:
                image = self.transform(image)
            return image, label
            
        def __len__(self):
            return len(self.dataset)
    
    # Apply transforms
    train_dataset = TransformDataset(train_dataset, transform=train_transforms)
    val_dataset = TransformDataset(val_dataset, transform=val_transforms)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader, dataset.classes

# Load pre-trained ResNet34 model
def load_model(num_classes):
    """Load and prepare ResNet34 model for fine-tuning"""
    model = models.resnet34(weights='IMAGENET1K_V1')
    
    # Freeze early layers
    for param in list(model.parameters())[:-8]:
        param.requires_grad = False
    
    # Replace the final fully connected layer
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    return model

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=10):
    """Train the model"""
    model = model.to(device)
    best_acc = 0.0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)
        
        # Validation phase
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(val_loader.dataset)
        epoch_acc = correct / total
        history['val_loss'].append(epoch_loss)
        history['val_acc'].append(epoch_acc)
        
        # Update learning rate
        scheduler.step()
        
        print(f'Epoch {epoch+1}/{num_epochs} | '
              f'Train Loss: {history["train_loss"][-1]:.4f} | '
              f'Train Acc: {history["train_acc"][-1]:.4f} | '
              f'Val Loss: {history["val_loss"][-1]:.4f} | '
              f'Val Acc: {history["val_acc"][-1]:.4f}')
        
        # Save best model
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(model.state_dict(), 'resnet34_plant_classifier.pth')
    
    print(f'Best validation accuracy: {best_acc:.4f}')
    return model, history

# Evaluate model
def evaluate_model(model, dataloader, class_names):
    """Evaluate model performance on a dataset"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Print classification report
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    # Create confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Plot confusion matrix
    plt.figure(figsize=(15, 15))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=90)
    plt.yticks(tick_marks, class_names)
    
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('confusion_matrix.png')
    plt.show()

# Plot training history
def plot_history(history):
    """Plot training and validation loss/accuracy"""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

# Function to predict a single image
def predict_image(model, image_path, class_names):
    """Predict the class of a single image"""
    model.eval()
    
    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    image_tensor = val_transforms(image).unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = nn.functional.softmax(output, dim=1)[0]
        _, predicted_idx = torch.max(output, 1)
        
    # Get top 5 predictions (or less if fewer classes)
    top_count = min(5, len(class_names))
    top_prob, top_idx = torch.topk(probabilities, top_count)
    
    print(f"Predicted class: {class_names[predicted_idx.item()]}")
    print(f"Top {top_count} predictions:")
    for i in range(top_count):
        print(f"{class_names[top_idx[i].item()]}: {top_prob[i].item()*100:.2f}%")
    
    # Display the image with prediction
    plt.figure(figsize=(8, 6))
    plt.imshow(image)
    plt.title(f"Prediction: {class_names[predicted_idx.item()]}")
    plt.axis('off')
    plt.show()

# Memory-efficient mixed precision training
def train_with_mixed_precision(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=10):
    """Train with mixed precision to save memory"""
    model = model.to(device)
    scaler = torch.cuda.amp.GradScaler()
    best_acc = 0.0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass with mixed precision
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            # Backward pass and optimize with scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)
        
        # Validation phase (same as before)
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(val_loader.dataset)
        epoch_acc = correct / total
        history['val_loss'].append(epoch_loss)
        history['val_acc'].append(epoch_acc)
        
        # Update learning rate
        scheduler.step()
        
        print(f'Epoch {epoch+1}/{num_epochs} | '
              f'Train Loss: {history["train_loss"][-1]:.4f} | '
              f'Train Acc: {history["train_acc"][-1]:.4f} | '
              f'Val Loss: {history["val_loss"][-1]:.4f} | '
              f'Val Acc: {history["val_acc"][-1]:.4f}')
        
        # Save best model
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(model.state_dict(), 'resnet34_plant_classifier.pth')
    
    print(f'Best validation accuracy: {best_acc:.4f}')
    return model, history

# Example usage
if __name__ == "__main__":
    # Set parameters
    data_dir = "/mnt/d/final dataset/train/"   # Replace with your dataset path (single folder)
    batch_size = 16  # Smaller batch size for 8GB VRAM
    num_epochs = 15
    learning_rate = 0.001
    train_split_ratio = 0.8  # 80% training, 20% validation
    
    # Create dataloaders with 80/20 split from a single folder
    train_loader, val_loader, class_names = create_dataloaders(
        data_dir, 
        batch_size=batch_size,
        train_ratio=train_split_ratio
    )
    
    # Get number of classes from class_names
    num_classes = len(class_names)
    print(f"Training on {num_classes} plant species.")
    
    # Load model
    model = load_model(num_classes)
    
    # Set up loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    # Train model with mixed precision for 8GB VRAM
    model, history = train_with_mixed_precision(
        model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs
    )
    
    # Plot training history
    plot_history(history)
    
    # Evaluate on validation set
    model.load_state_dict(torch.load('resnet34_plant_classifier.pth'))
    evaluate_model(model, val_loader, class_names)
    
    # Example of predicting a single image
    # predict_image(model, "path/to/test_image.jpg", class_names)