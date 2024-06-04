import os
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from tqdm import tqdm

from concurrent.futures import ThreadPoolExecutor


def generate_2d_slices(source_dir, corrupted_dir):
    source_2d = os.path.join(source_dir, "2d_slices_sagittal")
    corrupted_2d = os.path.join(corrupted_dir, "2d_slices_sagittal")

    os.makedirs(source_2d, exist_ok=True)
    os.makedirs(corrupted_2d, exist_ok=True)

    source_3d_images = sorted(map(str, Path(source_dir).glob("*nii")))
    corrupted_3d_images = sorted(map(str, Path(corrupted_dir).glob("*nii")))

    for source, corrupted in tqdm(
        zip(source_3d_images, corrupted_3d_images), total=len(source_3d_images)
    ):
        assert os.path.basename(source) == os.path.basename(corrupted)

        output_file_name = os.path.splitext(os.path.basename(source))[0]
        print(f"Processing: {source}")

        source_img = nib.load(source).get_fdata()
        corrupted_img = nib.load(corrupted).get_fdata()

        for i in range(source_img.shape[0]):
            source_slice_2d = source_img[i, :, :]
            corrupted_slice_2d = corrupted_img[i, :, :]

            if np.any(source_slice_2d != 0):
                plt.imsave(
                    f"{source_2d}/{output_file_name}_slice_{i}.png",
                    source_slice_2d,
                    cmap="gray",
                )
                plt.imsave(
                    f"{corrupted_2d}/{output_file_name}_slice_{i}.png",
                    corrupted_slice_2d,
                    cmap="gray",
                )


# For Afif
destination_gt_train_data_dir = "../assets/source_images_train"
destination_gt_val_data_dir = "../assets/source_images_val"

destination_motion_train_data_dir = "../assets/cc_motion_corrupted_train"
destination_motion_val_data_dir = "../assets/cc_motion_corrupted_val"

executor = ThreadPoolExecutor()

print("Generate slices for train data...")
executor.submit(generate_2d_slices, destination_gt_train_data_dir, destination_motion_train_data_dir)

print("Generate slices for validation data...")
executor.submit(generate_2d_slices, destination_gt_val_data_dir, destination_motion_val_data_dir)
