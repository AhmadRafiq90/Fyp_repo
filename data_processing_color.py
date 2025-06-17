from torchvision import transforms, datasets
from torch.utils.data import DataLoader

# Define transformations for data augmentation and normalization
transform = transforms.Compose([
    transforms.Resize((64, 64)),                      # Resize to 64x64
    transforms.RandomHorizontalFlip(),               # Apply random horizontal flip
    transforms.RandomRotation(10),                   # Random rotation within ±10 degrees
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Adjust colors
    transforms.ToTensor(),                            # Convert to tensor (should come before Normalize)
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# Load dataset
dataset = datasets.ImageFolder(root='output_data3/', transform=transform)

# Create a DataLoader
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class VideoDataset(Dataset):
    def __init__(self, root_dir, sequence_length=10, future_frames=5, transform=None):
        self.root_dir = root_dir
        self.sequence_length = sequence_length  # Total sequence length (input + future)
        self.future_frames = future_frames  # Number of future frames to predict
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))  # Class folders
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.video_paths = self.get_video_paths()

    def get_video_paths(self):
        video_paths = []
        for cls in self.classes:
            cls_folder = os.path.join(self.root_dir, cls)
            video_files = sorted(os.listdir(cls_folder))
            video_paths.extend([(os.path.join(cls_folder, vid), self.class_to_idx[cls]) for vid in video_files])
        return video_paths

    def __len__(self):
        return len(self.video_paths)

    def load_video_frames(self, video_folder):
        # Load the first 'sequence_length' frames
        frames = sorted(os.listdir(video_folder))[:self.sequence_length]
        frames = [os.path.join(video_folder, f) for f in frames]
        return frames

    def __getitem__(self, idx):
        video_folder, label = self.video_paths[idx]
        frame_paths = self.load_video_frames(video_folder)
        frames = [Image.open(frame_path) for frame_path in frame_paths]

        if self.transform:
            frames = [self.transform(frame) for frame in frames]

        # Stack frames into a tensor: [sequence_length, channels, height, width]
        frames = torch.stack(frames)

        # Split the frames into input and future frames
        inputs = frames[:-self.future_frames]  # First part as input (excluding the future frames)
        labels = frames[-self.future_frames:]  # Last part as the target (future frames)

        return inputs, labels  # Return input frames and future frames (targets)


# Define transforms for resizing and normalizing frames
transform = transforms.Compose([
    transforms.Resize((64, 64)),                      # Resize to 64x64
    transforms.RandomHorizontalFlip(),               # Apply random horizontal flip
    transforms.RandomRotation(10),                   # Random rotation within ±10 degrees
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Adjust colors
    transforms.ToTensor(),                            # Convert to tensor (should come before Normalize)
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])



# Load dataset
future_frames = 5  # Number of future frames to predict

# Create the dataset and data loader
train_dataset = VideoDataset(root_dir='output_data3/', sequence_length=10, future_frames=future_frames, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

