from dataclasses import dataclass, field
from pathlib import Path
import logging


@dataclass
class BaseConfig:
    log_level: int = logging.INFO


@dataclass
class YOLOConfig(BaseConfig):
    IMG_SIZE: int = 448

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
