import copy
from typing import Dict, Optional

import numpy as np
import torch
from loguru import logger
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from util import image_util


class Tester:
    def _save_images(self, images, names, output_path, **kwargs):
        for image, name, affine in zip(images, names, kwargs["affines"]):
            image = image.detach().cpu().numpy()
            image = image.squeeze()
            image_util.save_3d_image(image, output_path, name, affine)

    def _log_metrics(self, metric_scores):
        metric_str = [f"{key}: {value}" for key, value in metric_scores.items()]
        logger.info(f"Metric scores: {metric_str}")

    def test(
        self,
        model: nn.Module,
        test_dl: DataLoader,
        criterions: [Dict],
        device: str,
        data_output_path: str,
        metrics: [Dict],
    ):
        model.eval()
        test_loss = 0.0
        save_counter = 0
        mse_losses = []
        metric_scores = {}
        metric_eval = {}
        test_losses = {}
        with torch.no_grad():
            for (
                inputs,
                targets,
                image_filenames,
                label_filenames,
                image_affines,
                label_affines,
            ) in tqdm(test_dl, desc="Test step"):
                sum_loss = 0.0
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)

                for index, _ in enumerate(targets):
                    target = targets[index]
                    output = outputs[index]
                    mask = (target > 0).float()
                    filtered_output = output * mask
                    outputs[index] = filtered_output

                logger.info(f"Saving images: {image_filenames}")

                for index, _ in enumerate(image_filenames):
                    self._save_images(
                        [targets[index], inputs[index], outputs[index]],
                        [
                            f"{label_filenames[index]} Target",
                            f"{image_filenames[index]} Input",
                            f"{image_filenames[index]} Output",
                        ],
                        data_output_path,
                        affines=[
                            label_affines[index],
                            image_affines[index],
                            image_affines[index],
                        ],
                    )

                for loss_name, loss_fn in criterions.items():
                    loss = loss_fn(outputs, targets)
                    if loss_name == "mse_loss":
                        mse_losses.append(loss)

                    sum_loss += loss
                    test_losses[loss_name] = test_losses.get(loss_name, 0) + loss
                    test_loss += sum_loss.item()

                    for metric_name, metric_fn in metrics.items():
                        val = metric_fn(outputs, targets)
                        if metric_name not in metric_eval:
                            metric_eval[metric_name] = []
                        metric_eval[metric_name].append(val)

                        metric_fn.update(outputs, targets)

                save_counter += 1

        test_loss /= len(test_dl)
        print(test_losses)
        print(metric_eval)
        test_losses = {
            loss_name: loss / len(test_dl) for loss_name, loss in test_losses.items()
        }

        # Calculate standard deviation from metrics eval
        for metric_name, values in metric_eval.items():
            mean_value = torch.tensor(values).mean().item()
            std_value = torch.tensor(values).std().item()
            metric_scores["test_" + metric_name + "_mean"] = mean_value
            metric_scores["test_" + metric_name + "_std"] = std_value

        print(metric_scores)
