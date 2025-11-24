from dataclasses import dataclass
import logging


@dataclass
class Config:
    log_level: int = logging.INFO
