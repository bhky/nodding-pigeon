[![ci](https://github.com/bhky/head-gesture-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/bhky/head-gesture-detection/actions)
[![License MIT 1.0](https://img.shields.io/badge/license-MIT%201.0-blue.svg)](LICENSE)

# Introduction

The Head Gesture Detection (HGD) library provides a pre-trained model and 
a simple inference API for detecting head gestures in short videos.

# Installation

Tested for Python 3.8, 3.9, and 3.10.

The best way to install HGD with its dependencies is from PyPI:
```shell
python3 -m pip install --upgrade hgd
```
Alternatively, to obtain the latest version from this repository:
```shell
git clone git@github.com:bhky/head-gesture-detection.git
cd head-gesture-detection
python3 -m pip install .
```

# Quick test

An easy way to try out this library and the pre-trained model is to
make a short video with your head gesture.

The code snippet below will perform the following:
- Start webcam.
- End webcam when the needed number of frames has been collected (e.g., 60 frames).
- Make prediction of your head gesture and print the result to STDOUT.
```python
from hgd.inference import predict_video
# By default, the following call will download the pre-trained model weights 
# and start your webcam. You can press "q" to end recording before the auto end.
# The result is a dictionary.
result = predict_video()
print(result)
```
Result format:
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
The following `gesture` types are available:
- `nodding` - Repeatedly tilt your head upward and downward.
- `turning` - Repeatedly turn your head leftward and rightward.
- `stationary` - Not tilting or turning your head; translation motion is still treated as stationary.
- `undefined` - Unrecognised gesture.
