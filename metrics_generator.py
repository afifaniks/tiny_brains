import argparse
import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import DataLoader
from torchmetrics.image import (PeakSignalNoiseRatio,
                                StructuralSimilarityIndexMeasure,
                                VisualInformationFidelity)
from torchvision import transforms
from tqdm import tqdm

from dataset.custom_dataset import CustomDataset
from models.unet import UNet

def parse_arguments():
    parser = argparse.ArgumentParser(description="Script to evaluate model on test dataset")
    parser.add_argument("--test_image_dir", type=str, required=True, help="Path to the test image directory")
    parser.add_argument("--test_label_dir", type=str, required=True, help="Path to the test label directory")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the checkpoint")
    return parser.parse_args()

args = parse_arguments()

TRANSFORMATIONS = transforms.Compose([
    transforms.ToTensor(),
])

# Prepare dataset
logger.debug(f"Preparing datasets...")
test_dataset = CustomDataset(args.test_image_dir, args.test_label_dir, TRANSFORMATIONS)

# Training parameters
device = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16

# Data loaders
logger.debug(f"Preparing dataloaders...")
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

# Model
model = UNet()
model = model.to(device)
model.load_state_dict(torch.load(args.checkpoint))

# Metrics
metrics = {
    "psnr": PeakSignalNoiseRatio().to(device),
    "ssim": StructuralSimilarityIndexMeasure(data_range=1.0).to(device),
    "vif": VisualInformationFidelity().to(device)
}

# Test
model.eval()
val_loss = 0.0
scores = {}

with torch.no_grad():
    for inputs, targets in tqdm(test_loader, desc="Validation step"):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)

        # Compute loss
        criterion = nn.MSELoss()
        loss = criterion(outputs, targets)
        val_loss += loss.item() * inputs.size(0)

        # Compute metrics
        for metric_name, metric_fn in metrics.items():
            metric_fn.update(outputs, targets)

    val_loss /= len(test_loader.dataset)

    # Compute final metric scores
    scores = {"loss": val_loss}
    for metric_name, metric_fn in metrics.items():
        scores[metric_name] = metric_fn.compute().cpu().numpy().item()

print(scores)
