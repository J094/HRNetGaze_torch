# HRNFrameGaze_torch
A brand-new feature-based gaze estimation model for high resolution images.

## Requirement
```
pytorch (cuda)
tensorboard
numpy
opencv
imutils
dlib
```

## Example


## Webcam Demo
```
python demo_webcam.py
```
dlib models: https://github.com/davisking/dlib-models

I use dlib mmod_human_face_detector to detect face region.

Then, i use dlib shape_predictor_5_landmarks to get eye landmarks for clippling eye region.

After that, i use my HRNFrameGaze model to estimate gaze.
