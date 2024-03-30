import os
import shutil
import torch
import torch.nn as nn
from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import torch.optim as optim
from loguru import logger
import time

from dataset.custom_dataset import CustomDataset
from models.unet import UNet
from models.double_conv import DoubleConv
from util.trainer import Trainer


train_image_dir = "assets/train_images/train/data"
train_label_dir = "assets/train_images/train/gt"
validation_image_dir = "assets/train_images/val/data"
validation_label_dir = "assets/train_images/val/gt"

shutil.rmtree("assets/model_outputs")

os.mkdir("assets/model_outputs")

TRANSFORMATIONS = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                      std=[0.229, 0.224, 0.225] )
])

# Prepare dataset
logger.debug(f"Preparing datasets...")
train_dataset = CustomDataset(train_image_dir, train_label_dir, TRANSFORMATIONS)
test_dataset = CustomDataset(validation_image_dir, validation_label_dir, TRANSFORMATIONS)

# Training parameters
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# determine if we will be pinning memory during data loading
PIN_MEMORY = True if DEVICE == "cuda" else False

BATCH_SIZE = 16

# Data loaders
logger.debug(f"Preparing dataloaders...")
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=PIN_MEMORY, num_workers=0)
val_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=PIN_MEMORY, num_workers=0)

# Model
model = UNet()
model = model.to(DEVICE)

# Hyperparameters
lr = 5e-4

optimizer = optim.Adam(model.parameters(), lr=lr)
lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, threshold=5e-10)
criterion = nn.MSELoss()
epochs = 500
cur_time = int(time.time())
wandb_config = {
    "project": "tiny_brains",
    "name": f"unet_mri_{lr}_{cur_time}",
    "config": {
        "learning_rate": lr,
        "architecture": "U-Net",
        "dataset": "xx",
        "epochs": epochs,
    }
}

trainer = Trainer(wandb_config=wandb_config)

logger.debug(f"Starting training...")
trainer.train(
    model=model,
    epochs=epochs,
    optimizer=optimizer,
    scheduler=lr_scheduler,
    criterion=criterion,
    train_dl=train_loader,
    val_dl=val_loader,
    device=DEVICE,
    output_path="test.pth",
    early_stopping_patience=20
)

# model.load_state_dict(torch.load("test.pth"))
# model.eval()


