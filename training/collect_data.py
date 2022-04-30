"""
Collect landmark data from video files.
"""
from typing import Dict, List

import numpy as np

from dgd.video import video_to_landmarks


def main() -> None:
    video_path_dict = {
        "head_nodding": "",
        "head_turning": "",
        "undefined": "",
    }
    output_file_path = "landmarks.npz"

    landmark_dict: Dict[str, List[List[float]]] = {}
    for name, video_path in video_path_dict.items():
        landmark_dict[name] = video_to_landmarks(video_path)

    np.savez_compressed(output_file_path, **landmark_dict)


if __name__ == "__main__":
    main()
