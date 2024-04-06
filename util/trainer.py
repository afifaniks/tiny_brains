from typing import Optional

import torch
from loguru import logger
from torchvision.transforms import transforms
from tqdm import tqdm

from util.early_stopping import EarlyStopping
from util.wandb_manager import WandbManager


class Trainer:
    def __init__(self, wandb_config: Optional[dict] = None) -> None:
        self.wandb_manager = WandbManager(wandb_config) if wandb_config else None

    def train(self,
              *,
              model=None,
              epochs=None,
              optimizer=None,
              criterion=None,
              scheduler=None,
              train_dl=None,
              val_dl=None,
              device="cpu",
              output_path=None,
              early_stopping_patience: Optional[int] = None,
              metrics: Optional[dict] = None
              ):
        best_loss = float("inf")
        early_stopping = EarlyStopping(patience=early_stopping_patience, path=output_path) if early_stopping_patience else None

        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0.0
            for inputs, targets in tqdm(train_dl, desc="Training steps"):
                optimizer.zero_grad()
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * inputs.size(0)
            train_loss /= len(train_dl.dataset)

            # Validation
            model.eval()
            val_loss = 0.0
            metric_scores = {}
            all_validation_targets = []
            all_validation_preds = []

            with torch.no_grad():
                for inputs, targets in tqdm(val_dl, desc="Validation step"):
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)

                    if epoch % 5 == 0:
                        logger.info(f"Saving images at epoch: {epoch}")
                        self._save_images([targets[0], inputs[0], outputs[0]], [f"{epoch} target", f"{epoch} Input", f"{epoch} Output"])
                    loss = criterion(outputs, targets)
                    val_loss += loss.item() * inputs.size(0)
                    
                    all_validation_targets.extend(targets)
                    all_validation_preds.extend(outputs)

            val_loss /= len(val_dl.dataset)

            metric_scores["train_loss"] = train_loss
            metric_scores["val_loss"] = val_loss
            
            if metrics:
                for metric_name, metric_fn in metrics.items():                    
                    score = metric_fn(torch.stack(all_validation_preds), torch.stack(all_validation_targets))
                    metric_scores[metric_name] = score
            

            self._log_epoch(epochs, epoch, metric_scores)

            if early_stopping:
                early_stopping(val_loss=val_loss, model=model)

                if early_stopping.early_stop:
                    logger.debug("Early stopping...")
                    break

            elif val_loss < best_loss:
                best_loss = val_loss
                self._save_model(model, output_path)

            if scheduler:
                scheduler.step(val_loss)
                logger.debug(f"Current lr: {scheduler.get_last_lr()}")
        if self.wandb_manager:
            self.wandb_manager.finish()

    def _log_epoch(self, epochs, epoch, metric_scores):
        metric_str = [f"{key}: {value}" for key, value in metric_scores.items()]
        logger.info(f'Epoch [{epoch + 1}/{epochs}], {metric_str}')

        if self.wandb_manager:
            self.wandb_manager.log(metric_scores)

    def _save_model(self, model, output_path):
        logger.info("Saving new checkpoint...")
        torch.save(model.state_dict(), output_path)

    def _save_images(self, images, names):
        for image, name in zip(images, names):
            image = transforms.ToPILImage()(image)
            image.save("assets/model_outputs/{}.jpg".format(name))
