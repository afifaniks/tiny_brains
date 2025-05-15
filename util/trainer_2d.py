import copy
import os
from typing import List, Optional
from matplotlib import pyplot as plt

import torch
from loguru import logger
from torchvision.transforms import transforms
from tqdm import tqdm

from util.early_stopping import EarlyStopping
from util.wandb_manager import WandbManager


class Trainer2D:
    def __init__(self, wandb_config: Optional[dict] = None) -> None:
        self.wandb_manager = WandbManager(wandb_config) if wandb_config else None

    def train(
        self,
        *,
        model=None,
        epochs=None,
        optimizer=None,
        criterions=None,
        scheduler=None,
        train_dl=None,
        val_dl=None,
        device="cpu",
        model_output_path=None,
        data_output_path=None,
        early_stopping_patience: Optional[int] = None,
        metrics: Optional[dict] = None,
    ):
        best_loss = float("inf")
        early_stopping = (
            EarlyStopping(patience=early_stopping_patience, path=model_output_path)
            if early_stopping_patience
            else None
        )

        train_metrics = copy.deepcopy(metrics)
        val_metrics = copy.deepcopy(metrics)

        metric_scores = {}
        train_losses = {}
        val_losses = {}

        val_files = ["sub-CC00070XX05.nii", "sub-CC00065XX08.nii", "sub-CC00115XX08.nii"]

        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0.0
            train_losses = {}
            val_losses = {}
            for inputs, targets, image_filenames in tqdm(train_dl, desc="Training steps"):
                sum_loss = 0.0
                optimizer.zero_grad()
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)

                for loss_name, loss_fn in criterions.items():
                    loss = loss_fn(outputs, targets)
                    sum_loss = sum_loss + loss
                    train_losses[loss_name] = train_losses.get(loss_name, 0.0) + loss.item()

                sum_loss.backward()
                optimizer.step()
                train_loss = train_loss + sum_loss.item()

                for metric_name, metric_fn in train_metrics.items():
                    metric_fn.update(outputs, targets)

            train_loss /= len(train_dl)
            train_losses = {loss_name: loss / len(train_dl) for loss_name, loss in train_losses.items()}

            if metrics:
                for metric_name, metric_fn in train_metrics.items():
                    metric_scores["train_" + metric_name] = metric_fn.compute().cpu().numpy()
                for loss_name, loss in train_losses.items():
                    metric_scores["train_" + loss_name] = loss

            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, targets, image_filenames in tqdm(val_dl, desc="Validation step"):

                    sum_loss = 0.0
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)

                    if epoch % 5 == 0 and any(val_file in image_filename for image_filename in image_filenames for val_file in val_files):
                        logger.info(f"Saving images at epoch: {epoch}")
                        for index in range(len(inputs)):
                            self._save_images(
                                [targets[index], inputs[index], outputs[index]],
                                [
                                    f"{image_filenames[index]}_epoch{epoch}_target",
                                    f"{image_filenames[index]}_epoch{epoch}_input",
                                    f"{image_filenames[index]}_epoch{epoch}_output",
                                ],
                                output_dir=data_output_path
                            )
                    for loss_name, loss_fn in criterions.items():
                        loss = loss_fn(outputs, targets)
                        sum_loss = sum_loss + loss
                        val_losses[loss_name] = val_losses.get(loss_name, 0) + loss

                    val_loss = val_loss + sum_loss.item()

                    for metric_name, metric_fn in val_metrics.items():
                        metric_fn.update(outputs, targets)

            val_loss /= len(val_dl)
            val_losses = {loss_name: loss / len(val_dl) for loss_name, loss in val_losses.items()}

            if metrics:
                for metric_name, metric_fn in val_metrics.items():
                    metric_scores["val_" + metric_name] = metric_fn.compute().cpu().numpy()
                for loss_name, loss in val_losses.items():
                    metric_scores["val_" + loss_name] = loss

            metric_scores["train_loss"] = train_loss
            metric_scores["val_loss"] = val_loss

            metric_scores["train_loss"] = train_loss
            metric_scores["val_loss"] = val_loss

            self._log_epoch(epochs, epoch, metric_scores)

            if metrics:
                for metric_name, metric_fn in train_metrics.items():
                    metric_fn.reset()
                for metric_name, metric_fn in val_metrics.items():
                    metric_fn.reset()

            if early_stopping:
                early_stopping(val_loss=val_loss, model=model)

                if early_stopping.early_stop:
                    logger.debug("Early stopping...")
                    break

            elif val_loss < best_loss:
                best_loss = val_loss
                self._save_model(model, model_output_path)

            if scheduler:
                scheduler.step(val_loss)
                logger.debug(f"Current lr: {scheduler.get_last_lr()}")
        if self.wandb_manager:
            self.wandb_manager.finish()

    def _log_epoch(self, epochs, epoch, metric_scores):
        metric_str = [f"{key}: {value}" for key, value in metric_scores.items()]
        logger.info(f"Epoch [{epoch + 1}/{epochs}], {metric_str}")

        if self.wandb_manager:
            self.wandb_manager.log(metric_scores)

    def _save_model(self, model, output_path):
        logger.info("Saving new checkpoint...")
        torch.save(model.state_dict(), output_path)

    def _save_images(self, images, names, output_dir):
        for image, name in zip(images, names):
            image = image.squeeze()
            output_image = image.detach().cpu().numpy()
            plt.imsave(os.path.join(output_dir, "{}.png".format(name)), output_image, cmap="gray")
