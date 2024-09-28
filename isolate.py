import os
import shutil
import sys
import random
NUM_FOLDERS = 11
def move_large_folders(src_dir, dest_dir, threshold=150):
    if not os.path.exists(dest_dir):
        print("Directory does not exist")
        sys.exit(1)
    i = 1
    for folder_name in os.listdir(src_dir):
        if i == NUM_FOLDERS:
            print(f"Copied {NUM_FOLDERS} folders")
            break
        subdirectories = [d for d in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, d))]
        random.shuffle(subdirectories)
        folder_name = subdirectories.pop()
        folder_path = os.path.join(src_dir, folder_name)
        if os.path.isdir(folder_path):
            try:
                num_images = int(folder_name.split('(')[1].split(')')[0])
                if num_images > threshold:
                    dest_folder_path = os.path.join(dest_dir, folder_name)
                    if os.path.exists(dest_folder_path):
                        print(f"Skipping: {folder_name} (Already exists in destination)")
                    else:
                        shutil.copytree(folder_path, dest_folder_path)
                        print(f"Copied: {folder_name}")
                        i += 1
            except (IndexError, ValueError):
                print(f"Skipping: {folder_name} (Invalid format)")

# Example usage
src_directory = 'E:/Final Year Project/Dataset/PlantCLEF2024/train/'
dest_directory = 'E:/Final Year Project/Dataset/Selected_Species_Train/'
move_large_folders(src_directory, dest_directory)