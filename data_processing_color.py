# import cv2
# import os

# def ensure_dir(directory):
#     """Ensure that a directory exists. Create it if it doesn't."""
#     if not os.path.exists(directory):
#         try:
#             os.makedirs(directory)
#         except PermissionError as e:
#             print(f"PermissionError: {e}")
#             return False
#         except OSError as e:
#             print(f"OSError: {e}")
#             return False
#     return True

# def extract_frames_from_video(video_path, output_dir, frame_size=(64, 64), convert_to_gray=False):
#     """
#     Extract frames from a single video, resize them, and optionally convert to grayscale.
    
#     Args:
#     - video_path: Path to the input video file.
#     - output_dir: Directory where the frames will be saved.
#     - frame_size: Tuple (width, height) for resizing frames.
#     - convert_to_gray: Boolean, if True convert frames to grayscale.
#     """
#     # Check if the output directory is accessible
#     if not ensure_dir(output_dir):
#         print(f"Skipping video due to permission issues: {video_path}")
#         return
    
#     # Open the video file
#     cap = cv2.VideoCapture(video_path)
#     frame_count = 0
    
#     # Check if the video opened successfully
#     if not cap.isOpened():
#         print(f"Error opening video file: {video_path}")
#         return
    
#     # Loop over frames
#     while cap.isOpened():
#         ret, frame = cap.read()  # Read a frame
#         if not ret:
#             break  # End of video
        
#         # Resize the frame
#         frame_resized = cv2.resize(frame, frame_size)
        
#         # # Convert to grayscale if required
#         # if convert_to_gray:
#         #     frame_resized = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
        
#         # Save the frame in the output directory
#         output_frame_path = os.path.join(output_dir, f"frame_{frame_count:04d}.jpg")
#         cv2.imwrite(output_frame_path, frame_resized)
#         frame_count += 1
    
#     # Release the video capture object
#     cap.release()

# def extract_frames_from_folders(video_folders, output_root, frame_size=(64, 64), convert_to_gray=False):
#     """
#     Loop through multiple folders containing videos, extract frames from each video, 
#     and save them in corresponding output directories.
    
#     Args:
#     - video_folders: List of folder paths, each containing videos of a particular class.
#     - output_root: Root directory where all frames will be saved, organized by class.
#     - frame_size: Tuple (width, height) for resizing frames.
#     - convert_to_gray: Boolean, if True convert frames to grayscale.
#     """
#     for folder in video_folders:
#         # Get the class name from the folder path (e.g., 'HighJump', 'WalkingWithDog')
#         class_name = os.path.basename(folder.rstrip('/'))
#         class_output_dir = os.path.join(output_root, class_name)
#         #class_output_dir = os.path.join(class_output_dir, 'train/')
#         # Ensure the class output directory exists
#         if not ensure_dir(class_output_dir):
#             print(f"Skipping class {class_name} due to permission issues.")
#             continue
        
#         # Loop over all video files in the folder
#         for video_file in os.listdir(folder):
#             video_path = os.path.join(folder, video_file)
            
#             if video_file.endswith('.mp4') or video_file.endswith('.avi'):  # Check for video files
#                 video_output_dir = os.path.join(class_output_dir, os.path.splitext(video_file)[0])
                
#                 # Create output directory for the frames of each video
#                 if not ensure_dir(video_output_dir):
#                     print(f"Skipping video {video_file} due to permission issues.")
#                     continue
                
                
#                 # Extract frames from the current video and save them in the respective folder
#                 extract_frames_from_video(video_path, video_output_dir, frame_size, convert_to_gray)

# # Example usage
# video_paths = [
#     "D:/university/deep learning/project/dataset/train/HighJump/",
#     "D:/university/deep learning/project/dataset/train/WalkingWithDog/",
#     "D:/university/deep learning/project/dataset/train/Diving/",
#     "D:/university/deep learning/project/dataset/train/Biking/",
#     "D:/university/deep learning/project/dataset/train/JumpRope/"
# ]

# output_root = "D:/university/deep learning/project/output_data3/"
# extract_frames_from_folders(video_paths, output_root, frame_size=(64, 64), convert_to_gray=False)



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

