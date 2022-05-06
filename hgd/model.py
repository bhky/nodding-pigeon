"""
Model utilities.
"""
import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import numpy.typing
import tensorflow as tf
from tensorflow.keras import layers  # pylint: disable=import-error
from tensorflow.keras.models import Model  # pylint: disable=import-error,no-name-in-module

from hgd.config import Config
from hgd._download import download_weights_to, get_default_weights_path

NDFloat32Array = np.typing.NDArray[np.float32]


def make_model(weights_path: Optional[str] = get_default_weights_path()) -> Model:
    seq_input = layers.Input(
        shape=(Config.seq_length, Config.num_features),
        dtype=tf.float32, name="input"
    )
    # Shape: (batch_size, seq_length, num_features)
    x = seq_input
    x = layers.Conv1D(2, 10, strides=2, padding="valid", activation="relu")(x)
    x = layers.Conv1D(1, 5, strides=2, padding="valid", activation="relu")(x)
    x = layers.Flatten()(x)
    x_0 = layers.Dropout(0.01)(x)

    has_motion = layers.Dense(1, activation="sigmoid", name="has_motion")(x_0)
    gesture_probs = layers.Dense(len(Config.gesture_labels),
                                 activation="softmax", name="gesture_probs")(x_0)
    output = layers.Concatenate()([has_motion, gesture_probs])

    model = Model(seq_input, output)

    if weights_path is not None:
        if not os.path.isfile(weights_path):
            download_weights_to(weights_path)
        model.load_weights(weights_path)

    return model


def load_landmarks(npz_path: str) -> Dict[str, List[List[float]]]:
    loaded = np.load(npz_path)
    return {label: loaded[label].tolist() for label in loaded.files}


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
