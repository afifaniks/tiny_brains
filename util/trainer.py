from typing import Optional

from loguru import logger
from tqdm import tqdm

from util.wandb_manager import WandbManager


class Trainer:
    def __init__(self, wandb_config: Optional[dict] = None) -> None:
         if wandb_config:
            self.wandb_manager = WandbManager(**wandb_config)

    def train(self,
              *, 
              model=None,
              epochs=None, 
              optimizer=None,
              criterion=None, 
              train_dl=None, 
              val_dl=None, 
              device="cpu"
        ):
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
                train_loss /= len(train_loader.dataset)

                # Validation
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for inputs, targets in tqdm(val_dl, desc="Validation step"):
                        inputs, targets = inputs.to(device), targets.to(device)
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                        val_loss += loss.item() * inputs.size(0)
                val_loss /= len(val_loader.dataset)

                logger.info(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss}, Val Loss: {val_loss}')
                if self.wandb_manager:
                     self.wandb_manager.log({
                          "train_loss": {train_loss},
                          "val_loss": {val_loss}
                     })
        
