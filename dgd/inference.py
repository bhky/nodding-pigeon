"""
Inference utilities.
"""
import os
from typing import Optional, Sequence

from tensorflow.keras.models import Model

from dgd._download import get_default_weights_path, download_weights_to
from dgd.model import make_model
from dgd.video import video_to_landmarks


def load_pretrained_model(weights_path: str = get_default_weights_path()) -> Model:
    model = make_model()
    if not os.path.isfile(weights_path):
        download_weights_to(weights_path)
    model.load_weights(weights_path)
    return model


def predict_video(
        video_path: Optional[str] = None,  # None will start a webcam.
        model: Model = load_pretrained_model(),
        max_num_frames: int = 300,  # For the pre-trained model.
        padding: bool = True
) -> Sequence[float]:
    landmarks = video_to_landmarks(video_path, max_num_frames, padding)
    prediction: Sequence[float] = model.predict([landmarks])[0]
    return prediction
