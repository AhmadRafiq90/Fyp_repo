import os
import pandas as pd
import platform
import re

if platform.system() == 'Windows':
    path_start = "E:/"
elif platform.system() == 'Linux':
    path_start = '/media/ahmad/New Volume/'

"Loading the CSV File"

csv_file_path = path_start + "Final Year Project/Dataset/PlantCLEF2024/PlantCLEF2024singleplanttrainingdata.csv"
data = pd.read_csv(csv_file_path, delimiter=';')

"Root directory where images are stored i.e (train, test, val folders)"
root_directory = path_start + "Final Year Project/Dataset/PlantCLEF2024/train"

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
    species_id = str(row['species_id'])     
    species_name = str(row['species'])
    species_name = clean_string(species_name)

    "Construct old and new directory names"
    old_directory_name = os.path.join(root_directory, species_id)
    new_directory_name = os.path.join(root_directory, species_name)
    
    "If the old directory exsists then rename it"
    if os.path.isdir(old_directory_name):
        "If the directory has not already been renamed"
        if not os.path.exists(new_directory_name):
            os.rename(old_directory_name, new_directory_name)
            print(f"Renamed {old_directory_name} to {new_directory_name}")
            
print("Renaming complete.......................................")