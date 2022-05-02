"""
Data utilities.
"""
import time
from typing import Dict, List, Sequence

import numpy as np
import numpy.typing

from dgd.config import Config
from dgd.video import video_to_landmarks

NDFloat32Array = np.typing.NDArray[np.float32]


def collect_landmarks_with_webcam(
        labels: Sequence[str] = Config.labels,
        max_num_frames: int = 800,
        output_npz_path: str = Config.npz_filename,
        sleep_seconds: float = 3.0
) -> None:
    landmark_dict: Dict[str, List[List[float]]] = {}
    for label in labels:
        landmark_dict[label] = video_to_landmarks(None, max_num_frames)
        time.sleep(sleep_seconds)
    np.savez_compressed(output_npz_path, **landmark_dict)


def preprocess(features: NDFloat32Array) -> NDFloat32Array:
    # Smoothing.
    # https://danielmuellerkomorowska.com/2020/06/02/smoothing-data-by-rolling-average-with-numpy/
    kernel_size = 5
    kernel = np.ones(kernel_size) / kernel_size
    features = np.array(
        [np.convolve(row, kernel, mode="same") for row in features]
    )

    # Add stddev features.
    x_coord_std = np.std(features[:, 4:10], axis=1, keepdims=True)
    y_coord_std = np.std(features[:, 10:], axis=1, keepdims=True)
    features = np.concatenate([features, x_coord_std, y_coord_std], axis=1)

    assert features.shape[1] == Config.num_features
    return features


if __name__ == "__main__":
    collect_landmarks_with_webcam()
