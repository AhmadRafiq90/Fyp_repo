import os
import re

def count_subdirectories_with_threshold(directory, threshold):
    count = 0
    for subdir in os.listdir(directory):
        if os.path.isdir(os.path.join(directory, subdir)):
            match = re.search(r'\((\d+)\)$', subdir)
            if match:
                subdir_count = int(match.group(1))
                if subdir_count > threshold:
                    count += 1
    return count

# Example usage
directory_path = 'E:/Final Year Project/Dataset/PlantCLEF2024/train/'
threshold = 101
result = count_subdirectories_with_threshold(directory_path, threshold)
print(f"Number of subdirectories with count above {threshold}: {result}")