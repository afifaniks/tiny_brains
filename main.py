import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from loguru import logger

from dataset.custom_dataset import CustomDataset
from models.unet import UNet
from util.trainer import Trainer

train_image_dir = "assets/train_images/train/data"
train_label_dir = "assets/train_labels/train/gt"
validation_image_dir = "assets/train_images/val/data"
validation_label_dir = "assets/train_images/val/gt"

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
lr = 5e-5
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()
epochs = 30
wandb_config = {
    "project": "tiny_brains",
    "name": f"unet_mri_{lr}",
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
    criterion=criterion,
    train_dl=train_loader,
    val_dl=val_loader,
    device=DEVICE,
    output_path="test.pth"
)
