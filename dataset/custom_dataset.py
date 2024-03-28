import os
from PIL import Image

from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, image_dir, label_dir, transforms):
        self.image_dir = image_dir
        self.transforms = transforms

        self.image_paths = [os.path.join(image_dir, filename) for filename in os.listdir(image_dir) if
                            filename.endswith(".jpg")]
        self.label_paths = [os.path.join(label_dir, filename) for filename in os.listdir(label_dir) if
                            filename.endswith(".jpg")]

        if len(self.image_paths) != len(self.label_paths):
            raise Exception("Number of images and labels do not match")

    def __len__(self):
        # return the number of total samples contained in the dataset
        return len(self.image_paths)

    def __getitem__(self, idx):
        # grab the image path and label from the current index
        image_path = self.image_paths[idx]
        label_path = self.label_paths[idx]

        # load the image and label from disk, convert it to grayscale
        image = Image.open(image_path).convert('L')
        label = Image.open(label_path).convert('L')

        # check to see if we are applying any transformations
        if self.transforms is not None:
            # apply the transformations
            image = self.transforms(image)
            label = self.transforms(label)

        # return a tuple of the images
        return image, label
