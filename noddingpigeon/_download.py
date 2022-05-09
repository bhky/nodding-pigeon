"""
Download utilities.
"""

import os
from pathlib import Path

import gdown

from noddingpigeon import __weights_version__
from noddingpigeon.config import Config

WEIGHTS_URL = "https://github.com/bhky/nodding-pigeon/releases/download/" \
              f"weights_v{__weights_version__}/{Config.weights_filename}"


def _get_home_dir() -> str:
    return str(os.getenv("NODDING_PIGEON_HOME", default=Path.home()))


def get_default_weights_path() -> str:
    home_dir = _get_home_dir()
    return os.path.join(home_dir, f".noddingpigeon/weights/{Config.weights_filename}")


def download_weights_to(weights_path: str) -> None:
    download_dir = os.path.dirname(os.path.abspath(weights_path))
    os.makedirs(download_dir, exist_ok=True)
    print("Pre-trained weights will be downloaded.")
    gdown.download(WEIGHTS_URL, weights_path)
