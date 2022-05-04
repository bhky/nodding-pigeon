"""
Inference utilities.
"""
import os
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
from tensorflow.keras.models import Model

from hgd._download import get_default_weights_path, download_weights_to
from hgd.config import Config
from hgd.data import NDFloat32Array, preprocess
from hgd.model import make_model
from hgd.video import video_to_landmarks


def load_pretrained_model(weights_path: str = get_default_weights_path()) -> Model:
    model = make_model()
    if not os.path.isfile(weights_path):
        download_weights_to(weights_path)
    model.load_weights(weights_path)
    return model


def postprocess(
        prediction: Sequence[float],
        motion_threshold: float = 0.5,
        gesture_threshold: float = 0.9
) -> Dict[str, Any]:
    gesture_probs = prediction[1:]
    if prediction[0] < motion_threshold:
        label = Config.stationary_label
    else:
        if np.max(gesture_probs) < gesture_threshold:
            label = Config.undefined_gesture_label
        else:
            label = Config.gesture_labels[int(np.argmax(gesture_probs))]
    return {
        "gesture": label,
        "probabilities": {
            "has_motion": prediction[0],
            "gestures": {
                Config.gesture_labels[i]: gesture_probs[i]
                for i in range(len(Config.gesture_labels))
            }
        }
    }


def predict_video(
        video_path: Optional[str] = None,  # None will start a webcam.
        model: Optional[Model] = None,
        max_num_frames: int = Config.seq_length,  # For the pre-trained model.
        padding: bool = True,
        preprocess_fn: Callable[[List[List[float]]], NDFloat32Array] = preprocess
) -> Dict[str, Any]:
    if model is None:
        model = load_pretrained_model()
    landmarks = video_to_landmarks(video_path, max_num_frames, padding)
    prediction: Sequence[float] = model.predict(
        np.expand_dims(preprocess_fn(landmarks), axis=0)
    )[0].tolist()
    return postprocess(prediction)


# if __name__ == "__main__":
#     print(predict_video(model=load_pretrained_model(f"../training/{Config.weights_filename}")))
