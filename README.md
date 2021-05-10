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
yacs
```

## Example
![HRNFrameGazeJun01](https://z3.ax1x.com/2021/05/10/gtSzg1.gif)

![HRNFrameGazeJun02](https://z3.ax1x.com/2021/05/10/gtSx3R.gif)

## Webcam Demo
```
python demo_webcam.py
```
dlib models: https://github.com/davisking/dlib-models

I use dlib mmod_human_face_detector to detect face region.

Then, i use dlib shape_predictor_5_landmarks to get eye landmarks for clippling eye region.

After that, i use my HRNFrameGaze model to estimate gaze.
