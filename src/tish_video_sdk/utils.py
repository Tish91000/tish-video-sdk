import os
import shutil

def clean_folder(folder_path):
    """Deletes all files and subfolders inside the given folder path."""
    if not os.path.isdir(folder_path):
        print(f"Error: The path '{folder_path}' is not a valid directory.")
        return

    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        try:
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.unlink(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
        except Exception as e:
            print(f"Failed to delete {item_path}: {e}")

    print(f"Contents of '{folder_path}' have been cleared.")

def clean_folders(folders):
    """Cleans multiple folders by calling clean_folder on each."""
    for folder in folders:
        clean_folder(folder)