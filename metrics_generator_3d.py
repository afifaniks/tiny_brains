"""
Usage: python metrics_generator_3d.py --test_image_dir /work/disa_lab/projects/tiny_brains/cc_motion_corrupted_val/ --test_label_dir /work/disa_lab/projects/tiny_brains/source_images_val/ --runs 1 --checkpoint /home/mdafifal.mamun/research/tiny_brains/unet3d.pth
"""

import argparse

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import DataLoader
from torchmetrics.image import (
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
    VisualInformationFidelity,
)
from torchvision import transforms
from tqdm import tqdm

from dataset.nifti_dataset import NiftiDataset
from models.unet3d import UNet3D
from util.image_util import save_3d_image


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Script to evaluate model on test dataset"
    )
    parser.add_argument(
        "--test_image_dir",
        type=str,
        required=True,
        help="Path to the test image directory",
    )
    parser.add_argument(
        "--test_label_dir",
        type=str,
        required=True,
        help="Path to the test label directory",
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to the checkpoint"
    )
    parser.add_argument("--runs", type=int, default=5, help="Number of runs")
    return parser.parse_args()


def _save_images(images, names):
    model_output_path = "assets/model_outputs"
    for image, name in zip(images, names):
        save_3d_image(image, model_output_path, name)


args = parse_arguments()

TRANSFORMATIONS = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)

# Prepare dataset
image_shape = (256, 288, 288)
logger.debug(f"Preparing datasets...")
test_dataset = NiftiDataset(
    args.test_image_dir,
    args.test_label_dir,
    target_shape=image_shape,
    transform=TRANSFORMATIONS,
)
logger.debug(f"Number of samples: {len(test_dataset)}")

# Training parameters
device = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 2

# Data loaders
logger.debug(f"Preparing dataloaders...")
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
)

# Model
model = UNet3D()
model = model.to(device)

# Metrics
metrics = {
    "psnr": PeakSignalNoiseRatio().to(device),
    "ssim": StructuralSimilarityIndexMeasure(data_range=1.0).to(device),
    # "vif": VisualInformationFidelity().to(device),
}

num_runs = args.runs
all_scores = []

for run in range(num_runs):
    logger.debug(f"Run {run + 1}/{num_runs}")

    # Load model checkpoint
    model.load_state_dict(torch.load(args.checkpoint))

    # Test
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for step_idx, (inputs, targets) in tqdm(
            enumerate(test_loader), desc="Test step"
        ):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            # Compute loss
            criterion = nn.MSELoss()
            loss = criterion(outputs, targets)
            val_loss += loss.item() * inputs.size(0)

            # Compute metrics
            for metric_name, metric_fn in metrics.items():
                metric_fn.update(outputs, targets)

            if step_idx % 5 == 0:
                logger.info(f"Saving images at step: {step_idx}")
                _save_images(
                    [
                        targets[0].cpu().numpy(),
                        inputs[0].cpu().numpy(),
                        outputs[0].cpu().numpy(),
                    ],
                    [f"{step_idx} target", f"{step_idx} Input", f"{step_idx} Output"],
                )

        val_loss /= len(test_loader.dataset)

        # Compute final metric scores
        scores = {"loss": val_loss}
        for metric_name, metric_fn in metrics.items():
            scores[metric_name] = metric_fn.compute().cpu().numpy().item()

        all_scores.append(scores)

# Compute average and standard deviation
avg_scores = {}
std_scores = {}
for metric_name in metrics.keys():
    metric_values = [score[metric_name] for score in all_scores]
    avg_scores[metric_name] = np.mean(metric_values)
    std_scores[metric_name] = np.std(metric_values)

avg_loss = [score["loss"] for score in all_scores]
avg_scores["loss"] = np.mean(avg_loss)
std_scores["loss"] = np.std(avg_loss)

print("Average Scores:")
print(avg_scores)
print("Standard Deviation:")
print(std_scores)
