import os
import random
import shutil
import time

import torch
import torch.nn as nn
from loguru import logger
from torch import optim
from torch.utils.data import DataLoader
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchsummary import summary
from torchvision.transforms import transforms
from monai.losses import SSIMLoss

from dataset.nifti_dataset import NiftiDataset
from models.unet3d import UNet3D
from util.model_util import MaskedMSELoss, MaskedSsimLoss
from util.tester import Tester

random.seed(42)
torch.manual_seed(42)

# cur_time = int(time.time())

test_image_dir = "assets/test_2/corrupted"
test_label_dir = "assets/test_2/gt"

# test_image_dir = "assets/val_lvl_2/corrupted"
# test_label_dir = "assets/val_lvl_2/gt"

model_timestamp = "1740092881"

data_output_path = f"assets/model_outputs_tested_{model_timestamp}"

shutil.rmtree(data_output_path, ignore_errors=True)

os.mkdir(data_output_path)

TRANSFORMATIONS = transforms.Compose(
    [
        # transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                      std=[0.229, 0.224, 0.225] )
    ]
)

logger.info(f"Timestamp: {model_timestamp}")

# Prepare dataset
logger.debug(f"Preparing datasets...")
target_shape = (256, 288, 288)
# target_shape = (144, 184, 184)

test_dataset = NiftiDataset(test_image_dir, test_label_dir, target_shape, TRANSFORMATIONS)
logger.info(f"Test dataset size: {len(test_dataset)}")

# Training parameters
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# determine if we will be pinning memory during data loading
PIN_MEMORY = True if DEVICE == "cuda" else False

BATCH_SIZE = 1

# Data loaders
logger.debug(f"Preparing dataloaders...")
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    pin_memory=PIN_MEMORY,
    num_workers=0,
)

# Model
model = UNet3D()
# model.load_state_dict(torch.load("unet3d_1745344630.pth"))
# model.load_state_dict(torch.load("unet3d_1741214449.pth"))
model.load_state_dict(torch.load(f"unet3d_{model_timestamp}.pth"))
# model = Unet3DMonai()
model = model.to(DEVICE)

logger.debug(f'Model Summary: {summary(model, input_size=(1, 256, 288, 288), batch_size=BATCH_SIZE)}')
# logger.debug(f'Model Summary: {summary(model, input_size=(1, 144, 184, 184), batch_size=BATCH_SIZE)}')

# Hyperparameters
lr = 5e-4

# Metrics
metrics = {
    "psnr": PeakSignalNoiseRatio().to(DEVICE),
    "ssim": StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE),
    # "vif": VisualInformationFidelity().to(DEVICE),
}

#losses
mse_criterion = nn.MSELoss()
# masked_mse_criterion = MaskedMSELoss()
ssim_criterion = SSIMLoss(spatial_dims=3, data_range=1.0)
# ssim_criterion = CustomSsimLoss(data_range=1.0)
# residual_ssim_criterion = MaskedResidualSsimLoss(data_range=1.0)
# masked_ssim_criterion = MaskedSsimLoss(data_range=1.0)
# tv_criterion = TotalVariationLoss()
losses = {
    "ssim_loss": ssim_criterion,
    "mse_loss": mse_criterion,
    # "masked_ssim_loss": masked_ssim_criterion,
    # "residual_ssim_loss": residual_ssim_criterion,
    # "masked_mse_loss": masked_mse_criterion,
    # "tv": tv_criterion
}

tester = Tester()

logger.debug("Starting testing...")

tester.test(
    model=model,
    test_dl=test_loader,
    criterions=losses,
    device=DEVICE,
    data_output_path=data_output_path,
    metrics=metrics
)
