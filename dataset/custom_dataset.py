import os
from PIL import Image

from torch.utils.data import Dataset


class CustomDataset(Dataset):
	def __init__(self, image_dir, transforms):
		self.image_dir = image_dir
		self.transforms = transforms
		
		self.image_paths = [os.path.join(image_dir, filename) for filename in os.listdir(image_dir) if filename.endswith(".jpg")]
		
	def __len__(self):
		# return the number of total samples contained in the dataset
		return len(self.image_paths)
	
	def __getitem__(self, idx):
		# grab the image path from the current index
		image_path = self.image_paths[idx]
		
		# load the image from disk, convert it to grayscale
		image = Image.open(image_path).convert('L')
		
		# check to see if we are applying any transformations
		if self.transforms is not None:
			# apply the transformations
			image = self.transforms(image)
			
		# return a tuple of the images
		return image, image
    