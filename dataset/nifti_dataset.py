import os

import nibabel as nib
import numpy as np
import torchio as tio
from torch.utils.data import Dataset


class NiftiDataset(Dataset):
    def __init__(self, image_dir, label_dir, target_shape, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.target_shape = target_shape
        self.crop_or_pad = tio.CropOrPad(target_shape)

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

        # self.image_filenames = [
        #     os.path.join(image_dir, filename)
        #     for filename in os.listdir(image_dir)
        #     if filename.endswith((".nii", ".gz"))
        # ]
        # self.label_filenames = [
        #     os.path.join(label_dir, filename)
        #     for filename in os.listdir(label_dir)
        #     if filename.endswith((".nii", "gz"))
        # ]

    def __len__(self):
        return len(self.image_dir)

    def __getitem__(self, idx):
        image_filename = self.image_filenames[idx]
        label_filename = self.label_filenames[idx]

        image = nib.load(os.path.join(self.image_dir, image_filename))
        label = nib.load(os.path.join(self.label_dir, label_filename))

        image_affine = image.affine
        label_affine = label.affine

        # Resize image and mask to target shape
        image = self.crop_or_pad(image)
        label = self.crop_or_pad(label)

        image = image.get_fdata()
        label = label.get_fdata()

        image = (image - np.min(image)) / (np.max(image) - np.min(image))
        label = (label - np.min(label)) / (np.max(label) - np.min(label))

        # Convert to float32
        image = image.astype(np.float32)
        label = label.astype(np.float32)

        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)
            label = self.transform(label)

        return image.unsqueeze(0), label.unsqueeze(0), image_filename, label_filename, image_affine, label_affine
