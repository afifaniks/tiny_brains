import os

import numpy as np
import nibabel as nib
from PIL import Image


def save_2d_image(image: np.array, output_path: str, filename: str):
    pil_image = Image.fromarray(np.uint8(image * 255))
    pil_image.save(os.path.join(output_path, filename + ".jpg"))


def save_3d_image(image: np.array, output_path: str, filename: str, affine):
    nib_image = nib.Nifti1Image(image, affine=affine)
    nib.save(nib_image, os.path.join(output_path, filename + ".nii"))
