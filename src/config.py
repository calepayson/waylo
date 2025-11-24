from dataclasses import dataclass
import logging


@dataclass
class BaseConfig:
    log_level: int = logging.INFO


@dataclass
class CurrentConfig(BaseConfig):
    log_level: int = logging.DEBUG
