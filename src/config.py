import torch

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar
import logging


@dataclass
class BaseConfig:
    log_level: int = logging.INFO


@dataclass
class YOLOConfig(BaseConfig):
    img_size: int = 448

    split_size: int = 7
    n_boxes: int = 2
    n_classes: int = 20
    final_conv_channels: int = 1024

    lambda_coord: float = 5.0
    lambda_noobj: float = 0.5

    fc_hidden_size: int = 496  # 4096 in yolo paper but this speeds up training
    dropout: float = 0.0
    leaky_relu: float = 0.1

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    learning_rate: float = 3e-5
    weight_decay: float = 0

    epochs: int = 1000
    batch_size: int = 16
    num_workers: int = 2
    pin_memory: bool = True

    ARCHITECTURE: ClassVar[list] = [
        (7, 64, 2, 3),
        "M",
        (3, 192, 1, 1),
        "M",
        (1, 128, 1, 0),
        (3, 256, 1, 1),
        (1, 256, 1, 0),
        (3, 512, 1, 1),
        "M",
        [(1, 256, 1, 0), (3, 512, 1, 1), 4],
        (1, 512, 1, 0),
        (3, 1024, 1, 1),
        "M",
        [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
        (3, 1024, 1, 1),
        (3, 1024, 2, 1),
        (3, 1024, 1, 1),
        (3, 1024, 1, 1),
    ]

    data_root: Path = Path(__file__).parent.parent / "voc_data"

    @property
    def img_dir(self) -> Path:
        return self.data_root / "images"

    @property
    def label_dir(self) -> Path:
        return self.data_root / "labels"

    @property
    def train_csv(self) -> Path:
        return self.data_root / "train.csv"

    @property
    def val_csv(self) -> Path:
        return self.data_root / "val.csv"


@dataclass
class CurrentConfig(YOLOConfig):
    log_level: int = logging.DEBUG
