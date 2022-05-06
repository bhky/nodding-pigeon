"""
Train model for classifying landmark movements.
"""
import os
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import losses
from tensorflow.keras.models import Model

from hgd.config import Config
from hgd.data import NDFloat32Array, load_landmarks, preprocess
from hgd.model import make_model

tf.random.set_seed(0)


def setup_accelerators_and_get_strategy() -> tf.distribute.Strategy:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    # Strategy for GPU or multi-GPU machines.
    strategy = tf.distribute.MirroredStrategy()
    print(f"Number of accelerators: {strategy.num_replicas_in_sync}")
    return strategy


def make_ds_train(
        landmark_dict: Dict[str, Sequence[Sequence[float]]],
        seq_length: int,
        num_features: int,
        preprocess_fn: Callable[[Sequence[Sequence[float]]], NDFloat32Array],
        seed: int
) -> tf.data.Dataset:
    # Note: stationary label must come first in this design, see make_y.
    labels = (Config.stationary_label,) + Config.gesture_labels
    rng = np.random.default_rng(seed=seed)

    def make_y(label_idx: int) -> List[int]:
        has_motion = 1 if label_idx > 0 else 0
        y = [has_motion] + [0] * len(Config.gesture_labels)
        if has_motion == 1:
            y[label_idx] = 1
        # Note:
        # Format of y is [has_motion, <one-hot-gesture-class-vector>], e.g.,
        # [1, 0, 1, ..., 0] for the 2nd gesture,
        # [0, ..., 0] for stationary case.
        return y

    def gen() -> Tuple[List[List[float]], int]:
        while True:
            label_idx = int(rng.integers(len(labels), size=1))
            landmarks = landmark_dict[labels[label_idx]]
            seq_idx = int(rng.integers(len(landmarks) - seq_length, size=1))
            features = preprocess_fn(landmarks[seq_idx: seq_idx + seq_length])
            yield features, make_y(label_idx)

    return tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(seq_length, num_features), dtype=tf.float32),
            tf.TensorSpec(shape=(len(labels),), dtype=tf.int32)
        )
    )


def loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    has_motion_true = y_true[:, :1]
    has_motion_pred = y_pred[:, :1]
    has_motion_loss = losses.BinaryCrossentropy(
        reduction=tf.keras.losses.Reduction.NONE
    )(has_motion_true, has_motion_pred)

    # The class loss is designed in the way that, if has_motion is 0,
    # the box values do not matter.
    mask = y_true[:, 0] == 1
    weight = tf.where(mask, 1.0, 0.0)
    gesture_true = y_true[:, 1:]
    gesture_pred = y_pred[:, 1:]
    gesture_loss = losses.CategoricalCrossentropy(
        label_smoothing=0.05,
        reduction=tf.keras.losses.Reduction.NONE
    )(gesture_true, gesture_pred, sample_weight=weight)

    return (has_motion_loss + gesture_loss) * 0.5


class CustomAccuracy(tf.keras.metrics.Metric):

    def __init__(
            self,
            motion_threshold: float = 0.5,
            name: str = "custom_accuracy"
    ) -> None:
        super(CustomAccuracy, self).__init__(name=name)
        self.threshold = motion_threshold
        self.acc = tf.keras.metrics.CategoricalAccuracy()

    def update_state(
            self,
            y_true: tf.Tensor,
            y_pred: tf.Tensor,
            sample_weight: Optional[tf.Tensor] = None
    ) -> None:
        # IMPORTANT - the sample_weight parameter is needed to solve:
        # TypeError: tf__update_state() got an unexpected keyword argument 'sample_weight'
        y_pred = tf.where(y_pred[:, :1] >= self.threshold, y_pred, 0.0)
        self.acc.update_state(y_true[:, 1:], y_pred[:, 1:])

    def result(self) -> tf.Tensor:
        return self.acc.result()

    def reset_state(self) -> None:
        self.acc.reset_state()


def compile_model(model: Model) -> None:
    model.compile(
        loss=loss,
        optimizer=tf.keras.optimizers.Adam(amsgrad=True),
        metrics=[CustomAccuracy()],
    )


def train_and_save_weights(
        landmark_dict: Dict[str, List[List[float]]],
        model: Model,
        weights_path: str,
        preprocess_fn: Callable[[Sequence[Sequence[float]]], NDFloat32Array] = preprocess,
        seed: int = 42
) -> None:
    ds_train = make_ds_train(
        landmark_dict, Config.seq_length, Config.num_features,
        preprocess_fn, seed
    )
    ds_train = ds_train.batch(16).prefetch(tf.data.AUTOTUNE)

    # Kind of arbitrary here.
    mean_data_size = int(np.mean([len(v) for v in landmark_dict.values()]))
    steps_per_epoch = int(mean_data_size * 0.7)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=weights_path, monitor="loss", mode="min",
            save_best_only=True, save_weights_only=True, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="loss", min_delta=1e-04, patience=10,
            restore_best_weights=True, verbose=1
        ),
    ]
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
        compile_model(model)
    landmark_dict = load_landmarks(Config.npz_filename)
    try:
        train_and_save_weights(landmark_dict, model, Config.weights_filename)
    except KeyboardInterrupt:
        print("Training interrupted.")


if __name__ == "__main__":
    main()
