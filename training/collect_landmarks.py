"""
Landmark data collection.
"""
import os
import time
from typing import Sequence

import numpy as np

from hgd.video import video_to_landmarks
from hgd.config import Config
from hgd.model import load_landmarks


def _collect_landmarks_with_webcam(
        labels: Sequence[str] = (Config.stationary_label,) + Config.gesture_labels,
        npz_path: str = Config.npz_filename,
        max_num_frames: int = 800,
        sleep_seconds: float = 3.0,
        update_file: bool = True
) -> None:
    if os.path.isfile(npz_path) and update_file:
        landmark_dict = load_landmarks(npz_path)
    else:
        landmark_dict = {}

    for label in labels:
        landmark_dict[label] = video_to_landmarks(None, max_num_frames)
        time.sleep(sleep_seconds)
    np.savez_compressed(npz_path, **landmark_dict)


if __name__ == "__main__":
    _collect_landmarks_with_webcam()
