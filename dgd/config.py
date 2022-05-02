"""
Config.
"""
from dataclasses import dataclass
from typing import Sequence


@dataclass
class Config:
    labels: Sequence[str] = ("head_nodding", "head_turning")
    npz_filename: str = "landmarks.npz"
    seq_length: int = 60
    num_features: int = 18
    weights_filename: str = "dgd_weights.h5"
