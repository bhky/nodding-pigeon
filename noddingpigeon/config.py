"""
Config.
"""
from dataclasses import dataclass
from typing import Tuple

from noddingpigeon import __weights_version__


@dataclass
class Config:
    gesture_labels: Tuple[str, ...] = ("nodding", "turning")
    undefined_gesture_label: str = "undefined"
    stationary_label: str = "stationary"
    npz_filename: str = "landmarks.npz"
    seq_length: int = 60
    num_original_features: int = 16
    weights_filename: str = f"head_gesture_weights_v{__weights_version__}.h5"
