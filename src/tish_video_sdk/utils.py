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

import json
from pathlib import Path

def get_gemini_model(application_name, default_model="gemini-1.5-flash"):
    """
    Loads usage-specific Gemini model name from models_config.json.
    """
    # Search for models_config.json in parent directories
    current_path = Path(__file__).resolve()
    root_path = current_path

    # Traverse up nicely to find the config
    found = False
    for _ in range(5): # Check up to 5 levels up
        root_path = root_path.parent
        config_path = root_path / "models_config.json"
        if config_path.exists():
            found = True
            break

    if not found:
        # Fallback to hardcoded absolute path (specific to this user's requests)
        config_path = Path(r"C:\Users\cyril\Documents\Youtube Channels\Tish_Video_SDK\models_config.json")

    if not config_path.exists():
        print(f"Warning: models_config.json not found at {config_path}. Using default model: {default_model}")
        return default_model

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        return config.get(application_name, config.get("default", default_model))
    except Exception as e:
        print(f"Error loading models_config.json: {e}. Using default model.")
        return default_model
