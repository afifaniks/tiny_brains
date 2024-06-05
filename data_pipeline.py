import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np
import torchio as tio
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm

source_mask_dir = (
    "/work/disa_lab/projects/tiny_brains/unet_segmentation/unet_segmentation/WM-GM-CSF"
)
source_data_dir = (
    "/work/disa_lab/projects/tiny_brains/unet_segmentation/unet_segmentation/Images"
)
destination_root = "/work/disa_lab/projects/tiny_brains/final/"

assert len(os.listdir(source_data_dir)) == len(os.listdir(source_mask_dir))

augmented_source_destination_dir = os.path.join(
    destination_root, "augmented_source_images"
)
motion_corrupted_data_destination_dir = os.path.join(
    destination_root, "cc_motion_corrupted"
)

os.makedirs(augmented_source_destination_dir, exist_ok=True)
os.makedirs(motion_corrupted_data_destination_dir, exist_ok=True)

motion_transformation = tio.transforms.RandomMotion(
    degrees=5, translation=10, num_transforms=3
)


output_shape = (256, 288, 288)

crop_or_pad = tio.CropOrPad(output_shape)
source_images = os.listdir(source_data_dir)


def normalize_image_0_255(image, min_max=True):
    image_array = image.get_fdata()

    min_val = np.min(image_array)
    max_val = np.max(image_array)

    print(f"before normalization: {max_val}")

    if min_max:
        image_array = (image_array - min_val) / (max_val - min_val)
    else:
        image_array = image_array / 255

    print(f"after normalization: {np.max(image_array)}")

    normalized_image = nib.Nifti1Image(image_array, image.affine)

    return normalized_image


def apply_mask(image, mask):
    image_array = image.get_fdata()
    mask_array = mask.get_fdata()

    mask_array = mask_array.astype(bool)

    masked_image_array = image_array * mask_array

    masked_image = nib.Nifti1Image(masked_image_array, image.affine)

    return masked_image


def apply_contrast_inversion(image, mask):
    image_array = image.get_fdata()
    mask_array = mask.get_fdata()

    threshold = np.max(mask_array)

    binary_mask = mask_array == threshold

    max_intensity = np.max(image_array[binary_mask])

    inversed_image_array = image_array.copy()
    inversed_image_array[binary_mask] = max_intensity - image_array[binary_mask]

    print(inversed_image_array[:, :, 170][40, 150])

    inversed_image_array = (inversed_image_array - np.min(inversed_image_array)) / (
        np.max(inversed_image_array) - np.min(inversed_image_array)
    )

    inversed_image_array = (inversed_image_array * 255).astype(np.uint8)

    inversed_image = nib.Nifti1Image(inversed_image_array, image.affine)

    return inversed_image


def generate_2d_slices(source_dir, corrupted_dir):
    source_2d = os.path.join(source_dir, "2d_slices")
    corrupted_2d = os.path.join(corrupted_dir, "2d_slices")

    os.makedirs(source_2d, exist_ok=True)
    os.makedirs(corrupted_2d, exist_ok=True)

    source_3d_images = sorted(map(str, Path(source_dir).glob("*nii")))
    corrupted_3d_images = sorted(map(str, Path(corrupted_dir).glob("*nii")))

    for source, corrupted in tqdm(
        zip(source_3d_images, corrupted_3d_images), total=len(source_3d_images)
    ):
        assert os.path.basename(source) == os.path.basename(corrupted)

        output_file_name = os.path.splitext(os.path.basename(source))[0]
        tqdm.write(f"Processing: {source}")

        source_img = nib.load(source).get_fdata()
        corrupted_img = nib.load(corrupted).get_fdata()

        for i in range(source_img.shape[2]):
            tqdm.write(f"Processing: {source}\tGenerated Slice: {i+1}")
            source_slice_2d = source_img[:, :, i]
            corrupted_slice_2d = corrupted_img[:, :, i]

            if np.any(source_slice_2d != 0):
                plt.imsave(
                    f"{source_2d}/{output_file_name}_slice_{i}.png",
                    source_slice_2d,
                    cmap="gray",
                    vmin=0,
                    vmax=1,
                )
                plt.imsave(
                    f"{corrupted_2d}/{output_file_name}_slice_{i}.png",
                    corrupted_slice_2d,
                    cmap="gray",
                    vmin=0,
                    vmax=1,
                )


print("AUGMENT DATASET...")
for image in tqdm(source_images, total=len(source_images)):
    tqdm.write(f"Now processing: {image}")

    if image == ".DS_Store" or image == "CC0237_siemens_3_59_F.nii":
        continue

    start = time.time()

    three_d_image = nib.load(os.path.join(source_data_dir, image))
    image_mask = nib.load(os.path.join(source_mask_dir, image.split(".")[0] + ".nii"))

    three_d_image = normalize_image_0_255(three_d_image)
    three_d_image = apply_mask(three_d_image, image_mask)
    three_d_image = crop_or_pad(three_d_image)

    transformed_image = motion_transformation(three_d_image)
    transformed_image = normalize_image_0_255(transformed_image, min_max=True)

    nib.save(three_d_image, os.path.join(augmented_source_destination_dir, image))
    nib.save(
        transformed_image, os.path.join(motion_corrupted_data_destination_dir, image)
    )

    print(f"Time taken: {time.time() - start}")


print("\n\nCREATING SPLIT...")
all_images = [
    f for f in os.listdir(augmented_source_destination_dir) if f.endswith(".nii")
]
train_images, val_images = train_test_split(
    all_images,
    train_size=0.8,
    random_state=42,
    shuffle=True,
)

print("Train size", len(train_images), "Val size", len(val_images))
destination_gt_train_data_dir = os.path.join(augmented_source_destination_dir, "train")
destination_motion_train_data_dir = os.path.join(
    motion_corrupted_data_destination_dir, "train"
)
destination_gt_val_data_dir = os.path.join(augmented_source_destination_dir, "val")
destination_motion_val_data_dir = os.path.join(
    motion_corrupted_data_destination_dir, "val"
)

os.makedirs(destination_gt_train_data_dir, exist_ok=True)
os.makedirs(destination_motion_train_data_dir, exist_ok=True)
os.makedirs(destination_gt_val_data_dir, exist_ok=True)
os.makedirs(destination_motion_val_data_dir, exist_ok=True)

for image in tqdm(train_images, total=len(train_images), desc="Copying train data..."):
    if image == ".DS_Store":
        continue
    shutil.copy(
        os.path.join(augmented_source_destination_dir, image),
        os.path.join(destination_gt_train_data_dir, image),
    )
    shutil.copy(
        os.path.join(motion_corrupted_data_destination_dir, image),
        os.path.join(destination_motion_train_data_dir, image),
    )

for image in tqdm(val_images, total=len(val_images), desc="Copying val data..."):
    if image == ".DS_Store":
        continue
    shutil.copy(
        os.path.join(augmented_source_destination_dir, image),
        os.path.join(destination_gt_val_data_dir, image),
    )
    shutil.copy(
        os.path.join(motion_corrupted_data_destination_dir, image),
        os.path.join(destination_motion_val_data_dir, image),
    )

print("\n\nGENERATING 2D SLICES")
executor = ThreadPoolExecutor()

print("Generate slices for train data...")
executor.submit(
    generate_2d_slices, destination_gt_train_data_dir, destination_motion_train_data_dir
)

print("Generate slices for validation data...")
executor.submit(
    generate_2d_slices, destination_gt_val_data_dir, destination_motion_val_data_dir
)
