import os
import numpy as np

from PIL import Image
from torch.utils.data import Dataset
from loguru import logger

class CustomDataset(Dataset):
    def __init__(self, image_dir, label_dir, transforms, partial_dataset=False):
        self.image_dir = image_dir
        self.transforms = transforms
        self.lower_slice_limit = 170
        self.upper_slice_limit = 200

        self.image_paths = sorted(
            [
                os.path.join(image_dir, filename)
                for filename in os.listdir(image_dir)
                if filename.endswith(".png")
            ]
        )

        self.label_paths = sorted(
            [
                os.path.join(label_dir, filename)
                for filename in os.listdir(label_dir)
                if filename.endswith(".png")
            ]
        )

        if partial_dataset:
            self._filter_slices()

        if len(self.image_paths) != len(self.label_paths):
            raise Exception("Number of images and labels do not match")

    def _filter_slices(self):
        self.image_paths = [image_path for image_path in self.image_paths if self.lower_slice_limit <= self._get_slice_number(image_path) <= self.upper_slice_limit]
        self.label_paths = [label_path for label_path in self.label_paths if self.lower_slice_limit <= self._get_slice_number(label_path) <= self.upper_slice_limit]



    def _get_slice_number(self, image_name: str):
        last_underscore_split_string = image_name.split("_")[-1]
        return int(last_underscore_split_string.split(".")[0])

    def __len__(self):
        # return the number of total samples contained in the dataset
        return len(self.image_paths)

    def __getitem__(self, idx):
        # grab the image path and label from the current index
        image_path = self.image_paths[idx]
        label_path = self.label_paths[idx]

        file_name = os.path.splitext(os.path.basename(image_path))[0]

        # load the image and label from disk, convert it to grayscale
        image = Image.open(image_path).convert("L")
        label = Image.open(label_path).convert("L")

        image = np.array(image)
        label = np.array(label)

        # Normaize between 0 to 1
        # image = (image - np.min(image)) / (np.max(image) - np.min(image))
        # label = (label - np.min(label)) / (np.max(label) - np.min(label))

        # Normaize between 0 to 1 (when the image is in 0-255)
        image = image / 255
        label = label / 255

        # Convert to float32
        image = image.astype(np.float32)
        label = label.astype(np.float32)

        # check to see if we are applying any transformations
        if self.transforms is not None:
            # apply the transformations
            image = self.transforms(image)
            label = self.transforms(label)

        # return a tuple of the images
        return image, label, file_name
