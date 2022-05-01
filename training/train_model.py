"""
Train model for classifying landmark movements.
"""
import os
from typing import Dict, List

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import Model

from dgd.config import Config

tf.random.set_seed(0)


def setup_accelerators_and_get_strategy() -> tf.distribute.Strategy:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    # Strategy for GPU or multi-GPU machines.
    strategy = tf.distribute.MirroredStrategy()
    print(f"Number of accelerators: {strategy.num_replicas_in_sync}")
    return strategy


def load_landmarks(npz_file_path: str) -> Dict[str, List[List[float]]]:
    loaded = np.load(npz_file_path)
    return {label: loaded[label].tolist() for label in loaded.files}


def make_ds_train(
        landmark_dict: Dict[str, List[List[float]]],
        seq_length: int,
        num_features: int
) -> tf.data.Dataset:
    labels: List[str] = list(landmark_dict.keys())
    rng = np.random.default_rng(seed=42)

    def gen() -> List[List[float]]:
        label = rng.choice(labels, size=1)
        landmarks = landmark_dict[label]
        idx = rng.integers(len(landmarks) - seq_length, size=seq_length)
        yield landmarks[idx: idx + seq_length]

    return tf.data.Dataset.from_generator(
        gen,
        output_signature=tf.TensorSpec(
            shape=(seq_length, num_features), dtype=tf.float32
        )
    )


def make_model(
        seq_length: int,
        num_features: int,
        num_classes: int
) -> Model:
    seq_input = layers.Input(
        shape=(seq_length, num_features), dtype=tf.float32, name="input"
    )
    # Shape: (batch_size, seq_length, num_features)
    x = layers.LSTM(32, return_sequences=True)(seq_input)
    output = layers.Dense(num_classes, activation="softmax", name="output")(x)
    model = Model(seq_input, output)

    model.compile(
        loss="categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(),
        metrics=["acc"]
    )
    return model


def train_and_save_model(
        ds_train: tf.data.Dataset,
        model: Model,
        weights_path: str
) -> None:
    ds_train = ds_train.batch(32).prefetch(tf.data.AUTOTUNE)

    callbacks = [
        ModelCheckpoint(
            filepath=weights_path, monitor="loss", verbose=1,
            save_best_only=True, save_weights_only=False
        ),
        EarlyStopping(
            monitor="loss", min_delta=1e-08, patience=10, verbose=1,
            restore_best_weights=True
        ),
    ]
    model.fit(
        ds_train,
        epochs=500,
        callbacks=callbacks,
        verbose=1,
    )


def main() -> None:
    landmark_dict = load_landmarks(Config.npz_file_path)
    ds_train = make_ds_train(
        landmark_dict, Config.seq_length, Config.num_features
    )

    strategy = setup_accelerators_and_get_strategy()
    with strategy.scope():
        model = make_model(
            Config.seq_length, Config.num_features, Config.num_classes
        )

    train_and_save_model(ds_train, model, Config.model_path)


if __name__ == "__main__":
    main()
