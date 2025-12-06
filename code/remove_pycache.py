import os
import shutil

def delete_pycache_dirs(directory):
    for root, dirs, files in os.walk(directory):
        for dir_name in dirs:
            if dir_name == "__pycache__":
                dir_path = os.path.join(root, dir_name)
                try:
                    shutil.rmtree(dir_path)
                    print(f"Deleted: {dir_path}")
                except Exception as e:
                    print(f"Failed to delete {dir_path}: {e}")

if __name__ == "__main__":
    current_directory = os.getcwd()
    delete_pycache_dirs(current_directory)
    print("All __pycache__ directories have been deleted.")