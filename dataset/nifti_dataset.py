import nibabel as nib
import numpy as np
import torchio as tio
from skimage.transform import resize
from torch.utils.data import Dataset


class NiftiDataset(Dataset):
    def __init__(self, image_paths, mask_paths, target_shape, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.target_shape = target_shape
        self.crop_or_pad = tio.CropOrPad(target_shape)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = nib.load(self.image_paths[idx])
        mask = nib.load(self.mask_paths[idx])

        # Resize image and mask to target shape
        image = self.crop_or_pad(image)
        mask = self.crop_or_pad(mask)

        image = image.get_fdata()
        mask = mask.get_fdata()

        image = (image - np.min(image)) / (np.max(image) - np.min(image))
        mask = (mask - np.min(mask)) / (np.max(mask) - np.min(mask))

        # Convert to float32
        image = image.astype(np.float32)
        mask = mask.astype(np.float32)

        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image.unsqueeze(0), mask.unsqueeze(0)
