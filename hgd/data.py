"""
Data utilities.
"""
import os
import time
from typing import Dict, List, Sequence

import numpy as np
import numpy.typing

from hgd._video import video_to_landmarks
from hgd.config import Config

NDFloat32Array = np.typing.NDArray[np.float32]


def load_landmarks(npz_path: str) -> Dict[str, List[List[float]]]:
    loaded = np.load(npz_path)
    return {label: loaded[label].tolist() for label in loaded.files}


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


def preprocess(landmarks: Sequence[Sequence[float]]) -> NDFloat32Array:
    features = np.array(landmarks)

    # Make the landmarks relative to the face box.
    features[:, 4:10] = (features[:, 4:10] - features[:, 0:1]) / features[:, 2:3]
    features[:, 10:] = (features[:, 10:] - features[:, 1:2]) / features[:, 3:4]

    # Add stddev features.
    xs_coord_std = np.std(features[:, 4:10], axis=1, keepdims=True)
    ys_coord_std = np.std(features[:, 10:], axis=1, keepdims=True)
    features = np.concatenate([features, xs_coord_std, ys_coord_std], axis=1)

    # Drop face boxes.
    features = np.delete(features, [0, 1, 2, 3], axis=1)

    assert features.shape[1] == Config.num_features
    return features


if __name__ == "__main__":
    _collect_landmarks_with_webcam()
