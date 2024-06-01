import os

from PIL import Image
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, image_dir, label_dir, transforms):
        self.image_dir = image_dir
        self.transforms = transforms

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

        if len(self.image_paths) != len(self.label_paths):
            raise Exception("Number of images and labels do not match")

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

        # check to see if we are applying any transformations
        if self.transforms is not None:
            # apply the transformations
            image = self.transforms(image)
            label = self.transforms(label)

        # return a tuple of the images
        return image, label, file_name
