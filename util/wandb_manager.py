import os
import wandb

from dotenv import load_dotenv
from loguru import logger

load_dotenv()


class WandbManager:
    def __init__(self, lr: float, architecture: str, dataset: str, artifact: str, epochs: int) -> None:
        wandb.login(key=os.environ["WANDB_API_KEY"])
        logger.debug("Initializing wandb...")
        wandb.init(
            project="tiny_brains",
            name=artifact,
            config={
                "learning_rate": lr,
                "architecture": architecture,
                "dataset": dataset,
                "epochs": epochs,
            }    
        )
    
    def log(self, data: dict):
        wandb.log(data)

    def finish(self):
        logger.debug("Terminating wandb...")
        wandb.finish()

    