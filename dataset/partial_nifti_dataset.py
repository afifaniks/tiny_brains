import os
import nibabel as nib
import numpy as np
import torch
import torchio as tio
from loguru import logger
from torch.utils.data import Dataset


class PartialNiftiDataset(Dataset):
    def __init__(self, image_dir, label_dir, partial_files, target_shape=None, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.target_shape = target_shape
        self.partial_files = partial_files

        if self.target_shape:
            self.crop_or_pad = tio.CropOrPad(self.target_shape)

        self.image_filenames = [
            filename
            for filename in os.listdir(image_dir)
            if filename.endswith((".nii", ".gz"))
        ]
        self.label_filenames = [
            filename
            for filename in os.listdir(label_dir)
            if filename.endswith((".nii", "gz"))
        ]


    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_filename = self.image_filenames[idx]
        label_filename = image_filename

        for file in self.partial_files:
            if file in image_filename:
                image = nib.load(os.path.join(self.image_dir, image_filename))
                logger.debug(f"Loaded partial file {file}")
                break
        else:
            image = nib.load(os.path.join(self.label_dir, image_filename))

        # image = nib.load(os.path.join(self.image_dir, image_filename))
        label = nib.load(os.path.join(self.label_dir, label_filename))

        image_affine = image.affine
        label_affine = label.affine

        # Resize image and mask to target shape
        if self.target_shape:
            image = self.crop_or_pad(image)
            label = self.crop_or_pad(label)

        image = image.get_fdata()
        label = label.get_fdata()

        label_mask = (label > 0.01).astype(float)

        image = image * label_mask

        image = (image - np.min(image)) / (np.max(image) - np.min(image))
        label = (label - np.min(label)) / (np.max(label) - np.min(label))

        # Convert to float32
        image = image.astype(np.float32)
        label = label.astype(np.float32)

        # Manual Transform to Tensor
        image = torch.Tensor(image)
        label = torch.Tensor(label)

        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)
            label = self.transform(label)

        return image.unsqueeze(0), label.unsqueeze(0), image_filename, label_filename, image_affine, label_affine
