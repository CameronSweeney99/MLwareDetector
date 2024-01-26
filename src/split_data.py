#split_dataset
import os
import shutil
from sklearn.model_selection import train_test_split

def split_and_distribute_files(source_folder, dest_base_folder, split_ratio):
    # Make sure the base destination folder exists
    os.makedirs(dest_base_folder, exist_ok=True)

    # List all files in the source folder
    all_files = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]

    # Split the data
    train_files, test_files = train_test_split(all_files, test_size=split_ratio['test'], random_state=42)
    train_files, val_files = train_test_split(train_files, test_size=split_ratio['validation'] / (1 - split_ratio['test']), random_state=42)

    # Function to copy files to the designated folder
    def copy_files(files, type_folder):
        dest_folder = os.path.join(dest_base_folder, type_folder, os.path.basename(source_folder))
        os.makedirs(dest_folder, exist_ok=True)
        for file in files:
            shutil.copy(os.path.join(source_folder, file), os.path.join(dest_folder, file))

    # Copy the files to their respective subdirectories
    copy_files(train_files, 'training')
    copy_files(val_files, 'validation')
    copy_files(test_files, 'test')

# Define your source folders
benign_source_folder = 'E:/combined_dataset/benign'
malware_source_folder = 'E:/combined_dataset/malware'

# Define your base destination folder
dest_base_folder = 'E:/combined_dataset'

# Define your split ratios
split_ratio = {'training': 0.7, 'validation': 0.15, 'test': 0.15}

# Split and distribute the benign files
split_and_distribute_files(benign_source_folder, dest_base_folder, split_ratio)

# Split and distribute the malware files
split_and_distribute_files(malware_source_folder, dest_base_folder, split_ratio)
