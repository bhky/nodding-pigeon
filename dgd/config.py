"""
Config.
"""
from dataclasses import dataclass
from typing import Sequence


@dataclass
class Config:
    labels: Sequence[str] = ("head_nodding", "head_turning", "head_undefined")
    npz_filename: str = "landmarks.npz"
    seq_length: int = 100
    num_features: int = 12
    weights_filename: str = "dgd_weights.h5"
