"""
Config.
"""
from dataclasses import dataclass
from typing import Tuple


@dataclass
class Config:
    gesture_labels: Tuple[str, ...] = ("nodding", "turning")
    undefined_gesture_label: str = "undefined"
    stationary_label: str = "stationary"
    npz_filename: str = "landmarks.npz"
    seq_length: int = 60
    num_features: int = 14
    weights_filename: str = "hgd_weights.h5"
