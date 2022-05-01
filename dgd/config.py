"""
Config.
"""
from dataclasses import dataclass
from typing import Sequence


@dataclass
class Config:
    labels: Sequence[str] = ("head_nodding", "head_turning", "head_undefined")
    npz_file_path: str = "landmarks.npz"
    seq_length: int = 100
    num_features: int = 12
    model_path: str = "dgd_model.h5"
