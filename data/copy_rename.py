import os
import shutil
from tqdm import tqdm


def copy_files(source_folder, destination_folder, num_files=100):

    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    files = os.listdir(source_folder)

    files = files[:num_files]

    for i, file in tqdm(enumerate(files, start=1), total=len(files), desc="Copying files"):

        source_file_path = os.path.join(source_folder, file)
        file_extension = os.path.splitext(file)[1]
        new_file_name = f"{i}{file_extension}"
        destination_file_path = os.path.join(destination_folder, new_file_name)
        shutil.copy2(source_file_path, destination_file_path)


def rename_files(folder_path):
    files = os.listdir(folder_path)
    files = [f for f in files if os.path.isfile(os.path.join(folder_path, f))]
    files.sort()

    for i, file in tqdm(enumerate(files, start=1), total=len(files), desc="Rename files"):
        file_extension = os.path.splitext(file)[1]
        new_file_name = f"{i}{file_extension}"
        old_file_path = os.path.join(folder_path, file)
        new_file_path = os.path.join(folder_path, new_file_name)
        os.rename(old_file_path, new_file_path)


def main():
    source_folder = '/root/.cache/huggingface/coco/coco/bagon_coco2017_val_1000'
    destination_folder = '/root/.cache/huggingface/coco/coco/bagon_coco2017_val_10'

    # copy_files(source_folder, destination_folder, 10)

    rename_files(source_folder)

    print("Done.")


if __name__ == "__main__":
    main()