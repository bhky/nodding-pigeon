"""
Collect landmark data from video files.
"""
import time
from typing import Dict, List

import numpy as np

from dgd.video import video_to_landmarks


def main() -> None:
    labels = [
        "head_nodding",
        "head_turning",
        "head_undefined",
    ]
    output_file_path = "landmarks.npz"

    landmark_dict: Dict[str, List[List[float]]] = {}
    for label in labels:
        landmark_dict[label] = video_to_landmarks(None, 300)
        time.sleep(3.0)

    np.savez_compressed(output_file_path, **landmark_dict)


if __name__ == "__main__":
    main()
