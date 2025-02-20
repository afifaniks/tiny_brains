import os
import shutil
import time

import torch
import torch.nn as nn
import torch.optim as optim
import transformers
from loguru import logger
from monai.losses import SSIMLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchmetrics.image import (
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
    VisualInformationFidelity,
)
from torchvision import transforms
from torchsummary import summary

from dataset.nifti_dataset import NiftiDataset
from models.unet3d import UNet3D
from models.unet_monai import Unet3DMonai
from util.model_util import TotalVariationLoss, CustomSsimLoss, MaskedMSELoss, MaskedSsimLoss
from util.trainer import Trainer

cur_time = int(time.time())

train_image_dir = "assets/train/corrupted"
train_label_dir = "assets/train/gt"
validation_image_dir = "assets/val/corrupted"
validation_label_dir = "assets/val/gt"

data_output_path = f"assets/model_outputs_test_{cur_time}"

shutil.rmtree(data_output_path, ignore_errors=True)

os.mkdir(data_output_path)

TRANSFORMATIONS = transforms.Compose(
    [
        # transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                      std=[0.229, 0.224, 0.225] )
    ]
)

# Prepare dataset
logger.debug(f"Preparing datasets...")
target_shape = (256, 288, 288)
# target_shape = None

train_dataset = NiftiDataset(train_image_dir, train_label_dir, target_shape, TRANSFORMATIONS)
print(f"Train dataset size: {len(train_dataset)}")
val_dataset = NiftiDataset(
    validation_image_dir, validation_label_dir, target_shape, TRANSFORMATIONS
)
print(f"Test dataset size: {len(val_dataset)}")

# Training parameters
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# determine if we will be pinning memory during data loading
PIN_MEMORY = True if DEVICE == "cuda" else False

BATCH_SIZE = 2

# Data loaders
logger.debug(f"Preparing dataloaders...")
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    pin_memory=PIN_MEMORY,
    num_workers=0,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    pin_memory=PIN_MEMORY,
    num_workers=0,
)

# Model
# model = UNet3D()
model = Unet3DMonai()
model = model.to(DEVICE)

logger.debug(f'Model Summary: {summary(model, input_size=(1, 256, 288, 288), batch_size=BATCH_SIZE)}')

# Hyperparameters
lr = 1e-3

# Metrics
metrics = {
    "psnr": PeakSignalNoiseRatio().to(DEVICE),
    "ssim": StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE),
    # "vif": VisualInformationFidelity().to(DEVICE),
}

optimizer = optim.Adam(model.parameters(), lr=lr)
epochs = 200

# scheduler
# lr_scheduler = ReduceLROnPlateau(
#     optimizer, mode="min", factor=0.1, patience=10, threshold=5e-10
# )
# total_steps = len(train_loader) * epochs
# warmup_steps = int(0.025 * total_steps)

# logger.debug(f"Warmup steps: {warmup_steps}, Total Steps: {total_steps}")

# lr_scheduler = transformers.get_linear_schedule_with_warmup(
#     optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
# )

#losses
# mse_criterion = nn.MSELoss()
mse_criterion = MaskedMSELoss()
# ssim_criterion = SSIMLoss(spatial_dims=3, data_range=1.0)
ssim_criterion = CustomSsimLoss(data_range=1.0)
masked_ssim_criterion = MaskedSsimLoss(data_range=1.0)
# tv_criterion = TotalVariationLoss()
losses = {
    "ssim_loss": ssim_criterion,
    "masked_ssim_loss": masked_ssim_criterion,
    "mse_loss": mse_criterion
    # "tv": tv_criterion
}

wandb_config = {
    "project": "tiny_brains",
    "name": f"unet_adult_data_{lr}_3d_images_no_scheduler_{cur_time}",
    "config": {
        "learning_rate": lr,
        "architecture": "U-Net3d (16->128), loss: masked mse + ssim + masked ssim",
        "dataset": "Simulated Adult Image",
        "epochs": epochs,
        "changes": "Added masked ssim with original ssim and masked mse (data is lvl 3 motion)"
    },
}

# trainer = Trainer(wandb_config=wandb_config)
trainer = Trainer()

logger.debug("Starting training...")
trainer.train(
    model=model,
    epochs=epochs,
    optimizer=optimizer,
    scheduler=None,
    criterions=losses,
    train_dl=train_loader,
    val_dl=val_loader,
    device=DEVICE,
    model_output_path=f"unet3d_{cur_time}.pth",
    data_output_path=data_output_path,
    early_stopping_patience=20,
    metrics=metrics,
)

# model.load_state_dict(torch.load("test.pth"))
# model.eval()
