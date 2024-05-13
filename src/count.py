import os

def count_files_in_folder(directory):
    # Get a list of all files and directories in the specified directory
    try:
        entries = os.listdir(directory)
    except FileNotFoundError:
        print("The specified directory does not exist.")
        return None

    # Filter the list to include only files
    files = [entry for entry in entries if os.path.isfile(os.path.join(directory, entry))]
    
    # Count the number of files
    number_of_files = len(files)
    
    print(f"There are {number_of_files} files in the folder '{directory}'.")
    return number_of_files

# Example usage
directory_path = 'path/to/your/folder'  # Replace with the path to the directory you want to check
count_files_in_folder(directory_path)
