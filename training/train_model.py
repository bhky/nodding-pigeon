"""
Train model for classifying landmark movements.
"""
from typing import Dict, List

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import Model


def load_landmarks(npz_file_path: str) -> Dict[str, List[List[float]]]:
    loaded = np.load(npz_file_path)
    return {name: loaded[name].tolist() for name in loaded.files}


def make_ds_train(
        landmark_dict: Dict[str, List[List[float]]],
        seq_length: int,
        num_features: int
) -> tf.data.Dataset:
    names: List[str] = list(landmark_dict.keys())
    rng = np.random.default_rng(seed=42)

    def gen() -> List[List[float]]:
        name = rng.choice(names, size=1)
        landmarks = landmark_dict[name]
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
        verbose=2,
    )


def main() -> None:
    npz_file_path = "landmarks.npz"
    seq_length = 300  # That means the number of frames per "clip".
    num_features = 12  # From Mediapipe face detection.
    num_classes = 3
    model_path = "dgd_model.h5"

    landmark_dict = load_landmarks(npz_file_path)
    ds_train = make_ds_train(landmark_dict, seq_length, num_features)
    model = make_model(seq_length, num_features, num_classes)

    train_and_save_model(ds_train, model, model_path)


if __name__ == "__main__":
    main()
