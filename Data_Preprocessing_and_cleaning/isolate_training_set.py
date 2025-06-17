import os
import re
import shutil

def count_subdirectories_with_threshold(directory, threshold, copy_dir):
    count = 0
    for subdir in os.listdir(directory):
        if count == 12:
             break
        dr = os.path.join(directory, subdir)
        if os.path.isdir(dr):
            match = re.search(r'\((\d+)\)$', subdir)
            if match:
                subdir_count = int(match.group(1))
                if subdir_count >= threshold:
                    unique_copy_dir = os.path.join(copy_dir, subdir)
                    copy_directory(dr, unique_copy_dir)
                    print(dr)
                    count += 1
    return count

def copy_directory(src, dest):
    try:
            shutil.copytree(src, dest)
            print(f"Directory copied from {src} to {dest}")
    except Exception as e:
            print(f"Error: {e}")

# Example usage
directory_path = 'E:/Final Year Project/Dataset/PlantCLEF2024/train/'
copy_dir = 'E:/potential datasets/archive_2/plantvillage dataset/color/'
threshold = 200
result = count_subdirectories_with_threshold(directory_path, threshold, copy_dir)
print(f"Number of subdirectories with count above {threshold}: {result}")