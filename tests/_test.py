"""
Unit test.
"""
import os
import unittest

from noddingpigeon.config import Config
from noddingpigeon.inference import predict_video
from noddingpigeon.model import make_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NODDING_VIDEO_PATH = os.path.join(BASE_DIR, "head_nodding.mp4")
TURNING_VIDEO_PATH = os.path.join(BASE_DIR, "head_turning.mp4")
STATIONARY_VIDEO_PATH = os.path.join(BASE_DIR, "head_stationary.mp4")
TURNING_FAKE_VIDEO_PATH = os.path.join(BASE_DIR, "head_turning_fake.mp4")

MODEL_PATH = os.path.join(BASE_DIR, "head_stationary.mp4")

MODEL = make_model()
# For local testing only:
# MODEL = make_model(f"../training/{Config.weights_filename}")


def predict_gesture(video_path: str) -> str:
    result = predict_video(
        video_path=video_path,
        model=MODEL,
        from_beginning=False
    )
    gesture = result["gesture"]
    assert isinstance(gesture, str)
    return gesture


class TestModel(unittest.TestCase):

    def test_nodding(self):
        gesture = predict_gesture(NODDING_VIDEO_PATH)
        self.assertEqual(Config.gesture_labels[0], gesture)

    def test_turning(self):
        gesture = predict_gesture(TURNING_VIDEO_PATH)
        self.assertEqual(Config.gesture_labels[1], gesture)

    def test_stationary(self):
        gesture = predict_gesture(STATIONARY_VIDEO_PATH)
        self.assertEqual(Config.stationary_label, gesture)

    def test_turning_fake(self):
        gesture = predict_gesture(TURNING_FAKE_VIDEO_PATH)
        self.assertEqual(Config.stationary_label, gesture)


if __name__ == "__main__":
    unittest.main()
