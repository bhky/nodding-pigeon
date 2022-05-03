"""
Config.
"""
from dataclasses import dataclass
from typing import Tuple


@dataclass
class Config:
    class_labels: Tuple[str] = ("head_nodding", "head_turning")
    stationary_label: str = "stationary"
    npz_filename: str = "landmarks.npz"
    seq_length: int = 60
    num_features: int = 14
    weights_filename: str = "hgd_weights.h5"
