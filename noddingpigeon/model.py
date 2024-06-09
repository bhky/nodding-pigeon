"""
Model utilities.
"""
import os
from typing import Dict, List, Optional

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers  # pylint: disable=import-error,no-name-in-module
from tensorflow.keras.models import Model  # pylint: disable=import-error,no-name-in-module

from noddingpigeon.config import Config
from noddingpigeon._download import download_weights_to, get_default_weights_path


def make_model(weights_path: Optional[str] = get_default_weights_path()) -> Model:
    seq_input = layers.Input(
        shape=(Config.seq_length, Config.num_original_features),
        dtype=tf.float32, name="input"
    )
    # Shape: (batch_size, seq_length, num_features)
    x = seq_input
    x = Preprocessing()(x)
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


class Preprocessing(layers.Layer):  # type: ignore

    def __init__(self) -> None:
        super(Preprocessing, self).__init__(name="preprocessing")

    @staticmethod
    def call(
            x: tf.Tensor,
            training: Optional[bool] = None,
            mask: Optional[tf.Tensor] = None
    ) -> tf.Tensor:
        features = x

        # Make landmark features relative to the face box.
        xs_rel = (features[..., 4:10] - features[..., 0:1]) / features[..., 2:3]
        ys_rel = (features[..., 10:] - features[..., 1:2]) / features[..., 3:4]

        # Make stddev features.
        xs_coord_std = tf.math.reduce_std(xs_rel, axis=-1, keepdims=True)
        ys_coord_std = tf.math.reduce_std(ys_rel, axis=-1, keepdims=True)

        # Concatenate new features and exclude unwanted ones.
        features = tf.concat(  # pylint: disable=unexpected-keyword-arg,no-value-for-parameter
            [xs_rel, ys_rel, xs_coord_std, ys_coord_std], axis=-1
        )

        return features
