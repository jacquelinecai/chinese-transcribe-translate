import os
import shutil
import json

# Path to JSON file
json_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "hsk_characters", "character.json"))

# Load valid character folder names from JSON
def load_valid_characters(json_path):
    """
    Loads the list of valid Chinese characters from the JSON file.

    Args:
        json_path (str): The path to the JSON file.

    Returns:
        set: A set of valid Chinese characters.
    """
    if not os.path.exists(json_path):
        print(f"Error: JSON file '{json_path}' not found.")
        return set()

    try:
        with open(json_path, "r", encoding="utf-8") as file:
            data = json.load(file)
            return set(data.get("words", []))  # Convert list to a set for fast lookup
    except Exception as e:
        print(f"Error reading JSON file: {e}")
        return set()

# Get all folders in the target directory
def get_folders_from_target_directory(target_directory):
    """
    Retrieves a list of all folders directly under the given target directory.

    Args:
        target_directory (str): The directory path where to look for folders.

    Returns:
        list: A list of folder names found in the target directory.
    """
    if not os.path.isdir(target_directory):
        print(f"Directory '{target_directory}' does not exist.")
        return []

    return [entry.name for entry in os.scandir(target_directory) if entry.is_dir()]

# Delete folder if it's not in the valid list
def delete_unlisted_folders(target_directory, valid_names):
    """
    Deletes folders inside the target directory that are not in the valid_names set.

    Args:
        target_directory (str): The directory where folders exist.
        valid_names (set): A set of folder names that should be kept.
    """
    folders = get_folders_from_target_directory(target_directory)

    for folder in folders:
        if folder not in valid_names:
            folder_path = os.path.join(target_directory, folder)
            try:
                shutil.rmtree(folder_path)
                print(f"Deleted folder: {folder}")
            except Exception as e:
                print(f"Error deleting folder '{folder}': {e}")

# Main Execution
if __name__ == "__main__":
    # Get the absolute path to the "project" directory
    project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # Go up one level
    target_dir = os.path.join(project_dir, "data", "chinese_characters")  # Path to target directory

    # Load valid characters from JSON
    valid_characters = load_valid_characters(json_file_path)

    if valid_characters:
        print(f"Valid character folders: {len(valid_characters)} characters loaded.")
        delete_unlisted_folders(target_dir, valid_characters)
    else:
        print("No valid characters found. No folders will be deleted.")
