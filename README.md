![logo](logo/nodding-pigeon_logo.png)

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
# Example result:
# {'gesture': 'nodding',
#  'probabilities': {'has_motion': 1.0,
#   'gestures': {'nodding': 0.9576354622840881,
#    'turning': 0.042364541441202164}}}
```

## Video file

Alternatively, you could provide a pre-recorded video file:

```python
from noddingpigeon.inference import predict_video
from noddingpigeon.video import VideoSegment  # Optional.

result = predict_video(
  "your_head_gesture_video.mp4",
  video_segment=VideoSegment.LAST,  # Optionally change these parameters.
  motion_threshold=0.5,
  gesture_threshold=0.9
)
```
Note that no matter how long your video is, only the
pre-defined number of frames (`60` for the current model) are used for
prediction. The `video_segment` enum option controls how the frames 
are obtained from the video, 
e.g., `VideoSegment.LAST` means the last (`60`) frames will be used.

Thresholds can be adjusted as needed, see explanation in the 
[head gestures](#head-gestures) section.

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
- Otherwise, the largest probability from `gestures` is considered:
  - If it is smaller than `gesture_threshold` (default `0.9`), `gesture` is `undefined`,
  - else, the corresponding gesture label is selected (e.g., `nodding`).
- If no landmarks are detected in the video, `gesture` is `undefined`. 
  The `probabilities` dictionary is empty.

# API

## `noddingpigeon.inference`

### `predict_video`
Detect head gesture shown in the input video either from webcam or file.
- Parameters:
  - `video_path` (`Optional[str]`, default `None`): 
    File path to the video file, or `None` for starting a webcam.
  - `model` (`Optional[tf.keras.Model]`, default `None`): 
    A TensorFlow-Keras model instance, or `None` for using the default model.
  - `max_num_frames` (`int`, default `60`):
    Maximum number of frames to be processed by the model.
    Do not change when using the default model.    
  - `video_segment` (`VideoSegment` enum, default `VideoSegment.BEGINNING`):
    See explanation of [`VideoSegment`](#videosegment).
  - `end_padding` (`bool`, default `True`): 
    If `True` and `max_num_frames` is set, when the input video has not enough
    frames to form the feature tensor for the model, padding at the end will be 
    done using the features detected on the last frame.
  - `drop_consecutive_duplicates` (`bool`, default `True`):
    If `True`, features from a certain frame will not be used to form the 
    feature tensor if they are considered to be the same as the previous frame.
    This is a mechanism to prevent "fake" video created with static images.
  - `postprocessing` (`bool`, default `True`):
    If `True`, the final result will be presented as the Python dictionary
    described in the [usage](#usage) section, otherwise the raw model output
    is returned.
  - `motion_threshold` (`float`, default `0.5`):
    See the [head gestures](#head-gestures) section.
  - `gesture_threshold` (`float`, default `0.9`):
    See the [head gestures](#head-gestures) section.
- Return:
  - A Python dictionary if `postprocessing` is `True`, otherwise `List[float]`
    from the model output.

## `noddingpigeon.video`

### `VideoSegment`
Enum class for video segment options.
- `VideoSegment.BEGINNING`: Collect the required frames for the model from the beginning of the video.
- `VideoSegment.LAST`: Collect the required frames for the model toward the end of the video.

## `noddingpigeon.model`

### `make_model`
Create an instance of the model used in this library, 
optionally with pre-trained weights loaded.
- Parameters:
  - `weights_path` (`Optional[str]`, default `$HOME/.noddingpigeon/weights/*.h5`): 
    Path to the weights in HDF5 format to be loaded by the model. 
    The weights file will be downloaded if not exists.
    If `None`, no weights will be downloaded nor loaded to the model.
    Users can provide path if the default is not preferred. 
    The environment variable `NODDING_PIGEON_HOME` can also be used to indicate
    where the `.noddingpigeon/` directory should be located.
- Return:
  - `tf.keras.Model` object.
