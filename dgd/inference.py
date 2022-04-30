"""
Inference utilities.
"""
import os
from typing import Optional, Sequence

import tensorflow as tf
from tensorflow.keras.models import Model

from dgd._download import get_default_model_path, download_model_to
from dgd.video import video_to_landmarks


def load_model(model_path: str = get_default_model_path()) -> Model:
    if not os.path.isfile(model_path):
        download_model_to(model_path)
    model: Model = tf.keras.models.load_model(model_path)
    return model


def predict_video(
        video_path: Optional[str] = None,  # None will start a webcam.
        model: Optional[Model] = None,
        max_num_frames: int = 300,  # For the pre-trained model.
        padding: bool = True
) -> Sequence[float]:
    if model is None:
        model = load_model()
    landmarks = video_to_landmarks(video_path, max_num_frames, padding)
    prediction: Sequence[float] = model.predict([landmarks])[0]
    return prediction
