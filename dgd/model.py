"""
Model utilities.
"""
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model

from dgd.config import Config


def make_model(
        seq_length: int = Config.seq_length,
        num_features: int = Config.num_features,
        num_classes: int = len(Config.labels)
) -> Model:
    seq_input = layers.Input(
        shape=(seq_length, num_features), dtype=tf.float32, name="input"
    )
    # Shape: (batch_size, seq_length, num_features)
    x = layers.LSTM(1)(seq_input)
    output = layers.Dense(num_classes, activation="softmax", name="output")(x)
    model = Model(seq_input, output)
    return model
