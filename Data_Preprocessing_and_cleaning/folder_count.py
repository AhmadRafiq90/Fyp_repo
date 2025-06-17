import os
import re

def remove_existing_file_count(subdir):
        return re.sub(r'\(\d+\)$', '', subdir)

def append_file_count_to_subdirs(directory):
        for subdir in os.listdir(directory):
            subdir_path = os.path.join(directory, subdir)
            if os.path.isdir(subdir_path):
                clean_subdir_name = remove_existing_file_count(subdir)
                clean_subdir_path = os.path.join(directory, clean_subdir_name)
                if clean_subdir_path != subdir_path:
                    os.rename(subdir_path, clean_subdir_path)
                    subdir_path = clean_subdir_path
                file_count = sum([len(files) for r, d, files in os.walk(subdir_path)])
                new_subdir_name = f"{clean_subdir_name}({file_count})"
                new_subdir_path = os.path.join(directory, new_subdir_name)
                os.rename(subdir_path, new_subdir_path)

if __name__ == "__main__":
    directory = "E:/Final Year Project/Dataset/PlantCLEF2024/train/"
    append_file_count_to_subdirs(directory)

