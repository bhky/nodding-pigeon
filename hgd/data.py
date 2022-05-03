"""
Data utilities.
"""
import time
from typing import Dict, List, Sequence

import numpy as np
import numpy.typing

from hgd.config import Config
from hgd.video import video_to_landmarks

NDFloat32Array = np.typing.NDArray[np.float32]


def collect_landmarks_with_webcam(
        labels: Sequence[str] = (Config.stationary_label,) + Config.class_labels,
        max_num_frames: int = 800,
        output_npz_path: str = Config.npz_filename,
        sleep_seconds: float = 3.0
) -> None:
    landmark_dict: Dict[str, List[List[float]]] = {}
    for label in labels:
        landmark_dict[label] = video_to_landmarks(None, max_num_frames)
        time.sleep(sleep_seconds)
    np.savez_compressed(output_npz_path, **landmark_dict)


def preprocess(landmarks: List[List[float]]) -> NDFloat32Array:
    features = np.array(landmarks)

    # Scale with face box coordinates.
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
    collect_landmarks_with_webcam()
