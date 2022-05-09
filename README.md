[![ci](https://github.com/bhky/nodding-pigeon/actions/workflows/ci.yml/badge.svg)](https://github.com/bhky/nodding-pigeon/actions)
[![License MIT 1.0](https://img.shields.io/badge/license-MIT%201.0-blue.svg)](LICENSE)

# Introduction

The **Nodding Pigeon** library provides a pre-trained model and 
a simple inference API for detecting **head gestures** in short videos.
Under the hood, it uses Google [MediaPipe](https://google.github.io/mediapipe/)
for collecting the landmark features.

# Installation

Tested for Python 3.8, 3.9, and 3.10.

The best way to install this library with its dependencies is from PyPI:
```shell
python3 -m pip install --upgrade noddingpigeon
```
Alternatively, to obtain the latest version from this repository:
```shell
git clone git@github.com:bhky/nodding-pigeon.git
cd nodding-pigeon
python3 -m pip install .
```

# Usage

An easy way to try the API and the pre-trained model is to
make a short video with your head gesture.

## Webcam

The code snippet below will perform the following:
- Search for the pre-trained weights file from `$HOME/.noddingpigeon/weights/`,
  if not exists, the file will be downloaded from this repository.
- Start webcam.
- Collect the needed number of frames (default `60`) for the model.
- End webcam automatically (or you can press `q` to end earlier).
- Make prediction of your head gesture and print the result to STDOUT.
```python
from noddingpigeon.inference import predict_video

result = predict_video()
print(result)
```

## Video file

Alternatively, you could provide a pre-recorded video file:
```python
from noddingpigeon.inference import predict_video

result = predict_video(
  "your_head_gesture_video.mp4",
  from_beginning=False,
  motion_threshold=0.5,  # Optionally tune the thresholds.
  gesture_threshold=0.9
)
# The `from_beginning` flag controls whether the needed frames will be obtained
# from the beginning or toward the end of the video.
# Thresholds can be adjusted as needed, see explanation below.
```

## Result format

The result is returned as a Python dictionary.

```text
{
  'gesture': 'turning',
  'probabilities': {
    'has_motion': 1.0,
    'gestures': {
      'nodding': 0.009188028052449226,
      'turning': 0.9908120036125183
    }
  }
}
```

# Head gestures

The following `gesture` types are available:
- `nodding` - Repeatedly tilt your head upward and downward.
- `turning` - Repeatedly turn your head leftward and rightward.
- `stationary` - Not tilting or turning your head; translation motion is still treated as stationary.
- `undefined` - Unrecognised gesture or no landmarks detected (usually means no face is shown).

To determine the final `gesture`:
- If `has_motion` probability is smaller than `motion_threshold` (default `0.5`),
  `gesture` is `stationary`. Other probabilities are irrelevant.
- Otherwise, we will look for the largest probability from `gestures`:
  - If it is smaller than `gesture_threshold` (default `0.9`), `gesture` is `undefined`,
  - else, the corresponding gesture label is selected (e.g., `nodding`).
- If no landmarks are detected in the video, `gesture` is `undefined`. 
  The `probabilities` dictionary is empty.
