"""
Model utilities.
"""
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model

from hgd.config import Config


def make_model(
        seq_length: int = Config.seq_length,
        num_features: int = Config.num_features
) -> Model:
    seq_input = layers.Input(
        shape=(seq_length, num_features), dtype=tf.float32, name="input"
    )
    # Shape: (batch_size, seq_length, num_features)
    x = seq_input
    x = layers.Conv1D(2, 10, strides=5, padding="valid", activation="relu")(x)
    x = layers.Conv1D(1, 5, strides=2, padding="valid", activation="relu")(x)
    x = layers.Flatten()(x)
    x_0 = layers.Dropout(0.01)(x)

    has_motion = layers.Dense(1, activation="sigmoid", name="has_motion")(x_0)
    class_probs = layers.Dense(len(Config.class_labels),
                               activation="softmax", name="class_probs")(x_0)
    output = layers.Concatenate()([has_motion, class_probs])
    model = Model(seq_input, output)
    return model
