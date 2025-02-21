import os
import shutil

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

    folders = [entry.name for entry in os.scandir(target_directory) if entry.is_dir()]
    return folders

def delete_folder_if_matches(target_directory, target_folder_name):
    """
    Deletes a folder inside the target directory if its name matches the target.

    Args:
        target_directory (str): The directory to check.
        target_folder_name (str): The name of the folder to delete.
    """
    target_folder_path = os.path.join(target_directory, target_folder_name)  # Full path to target folder

    if os.path.isdir(target_folder_path):  # Check if the folder exists
        try:
            shutil.rmtree(target_folder_path)  # Delete folder and its contents
            print(f"Deleted folder: {target_folder_name} in {target_directory}")
        except Exception as e:
            print(f"Error deleting folder: {e}")
    else:
        print(f"Folder '{target_folder_name}' not found in {target_directory}.")

# Main Execution
if __name__ == "__main__":
    # Get the absolute path to the "project" directory
    project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # Go up one level
    target_dir = os.path.join(project_dir, "data", "smth")  # Path to "data/smth"

    folder_list = get_folders_from_target_directory(target_dir)

    if folder_list:
        print(f"Folders in '{target_dir}':")
        for folder in folder_list:
            print(folder)
            if folder == '心':  # Check if the folder matches the target name
                delete_folder_if_matches(target_dir, folder)
    else:
        print(f"No folders found in '{target_dir}'.")
