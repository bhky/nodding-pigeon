"""
Inference utilities.
"""
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Sequence

import numpy as np
from tensorflow.keras.models import Model  # pylint: disable=import-error,no-name-in-module

from noddingpigeon.video import video_to_landmarks
from noddingpigeon.config import Config
from noddingpigeon.model import make_model


@dataclass
class Result:
    gesture: str
    probabilities: Dict[str, Any]


def postprocess(
        prediction: Sequence[float],
        motion_threshold: float,
        gesture_threshold: float
) -> Dict[str, Any]:
    if not prediction:
        return asdict(Result(Config.undefined_gesture_label, {}))

    gesture_probs = prediction[1:]
    if prediction[0] < motion_threshold:
        label = Config.stationary_label
    else:
        if np.max(gesture_probs) < gesture_threshold:
            label = Config.undefined_gesture_label
        else:
            label = Config.gesture_labels[int(np.argmax(gesture_probs))]
    result = Result(
        gesture=label,
        probabilities={
            "has_motion": prediction[0],
            "gestures": {
                Config.gesture_labels[i]: gesture_probs[i]
                for i in range(len(Config.gesture_labels))
            }
        }
    )
    return asdict(result)


def predict_video(
        video_path: Optional[str] = None,  # None will start a webcam.
        model: Optional[Model] = None,
        max_num_frames: int = Config.seq_length,  # For the pre-trained model.
        from_beginning: bool = True,
        end_padding: bool = True,
        drop_consecutive_duplicates: bool = True,
        postprocessing: bool = True,
        motion_threshold: float = 0.5,
        gesture_threshold: float = 0.9
) -> Any:
    if model is None:
        model = make_model()
    landmarks = video_to_landmarks(
        video_path, max_num_frames, from_beginning, end_padding,
        drop_consecutive_duplicates
    )

    if landmarks:
        prediction: Sequence[float] = model.predict(
            np.expand_dims(landmarks, axis=0)
        )[0].tolist()
    else:
        prediction = []

    if not postprocessing:
        return prediction
    return postprocess(prediction, motion_threshold, gesture_threshold)


# if __name__ == "__main__":
#     print(predict_video(model=make_model(f"../training/{Config.weights_filename}")))
