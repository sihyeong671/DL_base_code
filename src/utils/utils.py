import os
import random
import numpy as np
from datetime import datetime
from dataclasses import dataclass, field

import torch


def get_current_time() -> str:
    """Return current time to str type"""
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")


@dataclass
class Config:
    """This is Config DataClass

    you can change hyperparameter in this class.
    """

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    epochs: int = 30
    batch_size: int = 8
    num_workers: int = 4
    learning_rate: float = 1e-4
    seed: int = 42
    data_dir: str = "data/"
    save_dir: str = "log/"
    exp_name: str = field(default_factory=get_current_time)
    mode: str = "train"
    model_name: str = "default"

    def save_config(self, path) -> None:
        # mlflow
        # tensorboard
        pass


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


if __name__ == "__main__":
    config = Config()
    print(config)
