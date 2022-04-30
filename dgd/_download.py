"""
Download utilities.
"""

import os
from pathlib import Path

import gdown  # type: ignore

MODEL_FILE = "dgd_model.h5"
MODEL_URL = f"https://github.com/bhky/dynamic-gesture-detection/releases/download/v0.1.0/{MODEL_FILE}"


def _get_home_dir() -> str:
    return str(os.getenv("DGD_HOME", default=Path.home()))


def get_default_model_path() -> str:
    home_dir = _get_home_dir()
    return os.path.join(home_dir, f".dgd/models/{MODEL_FILE}")


def download_model_to(model_path: str) -> None:
    download_dir = os.path.dirname(os.path.abspath(model_path))
    os.makedirs(download_dir, exist_ok=True)
    print(f"Pre-trained model will be downloaded.")
    gdown.download(MODEL_URL, model_path)
