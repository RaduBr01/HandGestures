import os

def count_elements_in_directories(start_path):
    for root, dirs, files in os.walk(start_path):
        # Count directories and files in the current directory
        total_elements = len(dirs) + len(files)
        print(f"Directory: {root}")
        print(f"  Total Elements: {total_elements} (Directories: {len(dirs)}, Files: {len(files)})")
        print("-" * 40)

if __name__ == "__main__":
    current_directory = "../training_data"
    print(f"Scanning from current directory: {current_directory}")
    print("=" * 40)
    count_elements_in_directories(current_directory)
