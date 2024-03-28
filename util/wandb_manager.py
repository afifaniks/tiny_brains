import os
import wandb

from dotenv import load_dotenv
from loguru import logger

load_dotenv()


class WandbManager:
    def __init__(self, config: dict) -> None:
        wandb.login(key=os.environ["WANDB_API_KEY"])
        logger.debug("Initializing wandb...")
        wandb.init(**config)
    
    def log(self, data: dict):
        wandb.log(data)

    def finish(self):
        logger.debug("Terminating wandb...")
        wandb.finish()

    