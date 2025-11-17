import os
import nibabel as nib
import torch
import io
import tarfile
from torch.utils.data import Dataset

class NiftiTarDataset(Dataset):
    def __init__(self, tar_path, transforms=None):
        self.tar_path = tar_path
        self.transforms = transforms

        # Open archive and list NIfTI files
        self.tar = tarfile.open(tar_path, 'r:gz')
        self.nifti_members = [m for m in self.tar.getmembers() if not "._" in str(m) and "/" in str(m) and ".DS_Store" not in str(m)]

    def __len__(self):
        return len(self.nifti_members)

    def __getitem__(self, idx):
        member = self.nifti_members[idx]
        print(f"Now extracting: {member}")

        # Extract file-like object into memory
        file_obj = self.tar.extractfile(member)
        if file_obj is None:
            raise RuntimeError(f"Failed to extract {member.name} from tar archive.")

        # Read NIfTI file from the in-memory buffer
        file_bytes = io.BytesIO(file_obj.read())
        img = nib.Nifti1Image.from_bytes(file_bytes.read())
        data = img.get_fdata()

        if self.transforms:
            data = self.transforms(data)
        else:
            data = torch.from_numpy(data).type(torch.float32)

        return data.unsqueeze(0), data.unsqueeze(0), member.name, member.name, img.affine, img.affine

    def __del__(self):
        # Close the tar file on cleanup
        if hasattr(self, 'tar') and self.tar:
            self.tar.close()