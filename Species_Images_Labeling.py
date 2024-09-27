import os
import pandas as pd
import platform
import re

if platform.system() == 'Windows':
    path_start = "G:/"
elif platform.system() == 'Linux':
    path_start = '/media/ahmad/New Volume/'

"Loading the CSV File"
csv_file_path = path_start + "Final Year Project/Dataset/PlantCLEF2024/PlantCLEF2024singleplanttrainingdata.csv"
data = pd.read_csv(csv_file_path, delimiter=';')

"Root directory where images are stored i.e (train, test, val folders)"

root_directory = path_start + "Final Year Project/Dataset/PlantCLEF2024/test"

"List of Organ types"
organ_types = ['fruit', 'flower', 'scan', 'branch', 'habit', 'bark', 'leaf']

"Initialize a dictionary to store counts for each organ type"
organ_counts = {organ: 0 for organ in organ_types}

"Specifically for linux since it doesn't allow special characters in filename"
def clean_string(input_string):
    """
    Removes any non-alphanumeric characters except for spaces and underscores from the input string.

    Args:
        input_string (str): The string to be cleaned.

    Returns:
        str: The cleaned string.
    """
    # Using regex to allow only alphanumeric characters, spaces, and underscores
    cleaned_string = re.sub(r'[^\w\s]', '', input_string)
    return cleaned_string

"Iterating through CSV"
for index, row in data.iterrows():
    "Extract specie_id(Directory) of the specie and it's name"
    species_image_name = str(row['image_name'])
    species_id = str(row['species_id'])
    species_name = str(row['species'])
    species_organ_type = str(row['organ'])

    species_image_name = clean_string(species_image_name)
    species_name = clean_string(species_name)
    
    organ_counts[species_organ_type] += 1
        
    "Construct the directory path for the species id"
    species_directory = os.path.join(root_directory, species_name)
    
    "If the species is found"
    if os.path.isdir(species_directory):
        "then go inside the folder to rename images"
        for file_name in os.listdir(species_directory):
            if file_name == species_image_name:
                "Construct Old and new file paths"
                old_file_path = os.path.join(species_directory, file_name)
                new_file_name = f"{species_name}_{species_organ_type}_{organ_counts[species_organ_type]}.jpg"
                new_file_path = os.path.join(species_directory, new_file_name)
                
                "Rename the image file"
                if not os.path.exists(new_file_path):
                    os.rename(old_file_path, new_file_path)
                    print(f"Renamed {old_file_path} to {new_file_path}")
    else:
        print(f"Directory {species_directory} not found")
            
print("Renaming complete.......................................")