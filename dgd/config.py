"""
Config.
"""
from dataclasses import dataclass


@dataclass
class Config:
    npz_file_path: str = "landmarks.npz"
    seq_length: int = 100
    num_features: int = 12
    num_classes: int = 3
    model_path: str = "dgd_model.h5"
