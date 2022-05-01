"""
Data utilities.
"""
import time
from typing import Dict, Sequence

import numpy as np

from dgd.config import Config
from dgd.video import video_to_landmarks


def collect_landmarks_with_webcam(
        labels: Sequence[str] = Config.labels,
        output_npz_file_path: str = Config.npz_file_path,
        sleep_seconds: float = 3.0
) -> None:
    landmark_dict: Dict[str, Sequence[Sequence[float]]] = {}
    for label in labels:
        landmark_dict[label] = video_to_landmarks(None, 800)
        time.sleep(sleep_seconds)
    np.savez_compressed(output_npz_file_path, **landmark_dict)


if __name__ == "__main__":
    collect_landmarks_with_webcam()
