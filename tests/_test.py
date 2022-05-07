"""
Unit test.
"""
import os
import unittest

from hgd.config import Config
from hgd.inference import predict_video
from hgd.model import make_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NODDING_VIDEO_PATH = os.path.join(BASE_DIR, "head_nodding.mp4")
TURNING_VIDEO_PATH = os.path.join(BASE_DIR, "head_turning.mp4")
STATIONARY_VIDEO_PATH = os.path.join(BASE_DIR, "head_stationary.mp4")
TURNING_FAKE_VIDEO_PATH = os.path.join(BASE_DIR, "head_turning_fake.mp4")

MODEL_PATH = os.path.join(BASE_DIR, "head_stationary.mp4")

# For local testing only:
# MODEL = make_model(f"../training/{Config.weights_filename}")
MODEL = make_model()


def predict_gesture(video_path: str) -> str:
    gesture = predict_video(
        video_path=video_path,
        model=MODEL,
        from_beginning=False
    )["gesture"]
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

    @unittest.skip("Not working yet.")
    def test_turning_fake(self):
        gesture = predict_gesture(TURNING_FAKE_VIDEO_PATH)
        self.assertEqual(Config.stationary_label, gesture)


if __name__ == "__main__":
    unittest.main()
