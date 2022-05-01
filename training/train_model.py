"""
Train model for classifying landmark movements.
"""
import os
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import Model

from dgd.config import Config
from dgd.model import make_model

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
        num_features: int,
        seed: int
) -> tf.data.Dataset:
    labels: List[str] = list(landmark_dict.keys())
    rng = np.random.default_rng(seed=seed)

    def to_categorical(idx: int) -> List[int]:
        y = [0] * len(labels)
        y[idx] = 1
        return y

    def gen() -> Tuple[List[List[float]], int]:
        while True:
            label_idx = int(rng.integers(len(labels), size=1))
            landmarks = landmark_dict[labels[label_idx]]
            seq_idx = int(rng.integers(len(landmarks) - seq_length, size=1))
            yield landmarks[seq_idx: seq_idx + seq_length], to_categorical(label_idx)

    return tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(seq_length, num_features), dtype=tf.float32),
            tf.TensorSpec(shape=(len(labels),), dtype=tf.int32)
        )
    )


def train_and_save_weights(
        landmark_dict: Dict[str, List[List[float]]],
        model: Model,
        weights_path: str,
        seed: int = 42
) -> None:
    ds_train = make_ds_train(
        landmark_dict, Config.seq_length, Config.num_features, seed
    )
    ds_train = ds_train.batch(16).prefetch(tf.data.AUTOTUNE)

    # Kind of arbitrary here.
    mean_data_size = int(np.mean([len(v) for v in landmark_dict.values()]))
    steps_per_epoch = int(mean_data_size * 0.7)

    callbacks = [
        ModelCheckpoint(
            filepath=weights_path, monitor="loss", verbose=1,
            save_best_only=True, save_weights_only=True
        ),
        EarlyStopping(
            monitor="loss", min_delta=1e-05, patience=5, verbose=1,
            restore_best_weights=True
        ),
    ]
    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.01),
        optimizer=tf.keras.optimizers.Adam(amsgrad=True),
        metrics=["acc"]
    )
    model.fit(
        ds_train,
        epochs=500,
        steps_per_epoch=steps_per_epoch,
        callbacks=callbacks,
        verbose=1,
    )


def main() -> None:
    strategy = setup_accelerators_and_get_strategy()
    with strategy.scope():
        model = make_model()
    landmark_dict = load_landmarks(Config.npz_filename)
    try:
        train_and_save_weights(landmark_dict, model, Config.weights_filename)
    except KeyboardInterrupt:
        print("Training interrupted.")


if __name__ == "__main__":
    main()
