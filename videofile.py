from noddingpigeon.inference import predict_video
from noddingpigeon.video import VideoSegment  # Optional.

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("name", help="video file name")
args = parser.parse_args()
videoname = args.name
result = predict_video(
  videoname,
  video_segment=VideoSegment.LAST,  # Optionally change these parameters.
  motion_threshold=0.5,
  gesture_threshold=0.9
)
print(result)
