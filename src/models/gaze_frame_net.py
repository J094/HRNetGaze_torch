import logging
import os

import torch
import torch.nn as nn
import cv2 as cv
import numpy as np

from src.configs import cfg


BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


def get_gaze_frame(heatmaps, landmarks):
    """
    Input: heatmaps, landmarks (torch.tensor in gpu) (n, c, h, w)
    Output: gaze_frame combined with heatmaps and landmarks (torch.tensor in gpu) (n, 1, h, w)
    """
    heatmaps = heatmaps.cpu().detach().numpy()
    landmarks = landmarks.cpu().detach().numpy()

    n, c, h, w = heatmaps.shape
    landmarks = np.floor((landmarks + 0.5), dtype=np.float)
    landmarks = np.int32(landmarks)
    frames = np.zeros((n, h, w))
    for i in range(n):
        heatmap = np.zeros(cfg.MODEL.IMAGE_SIZE)
        frame_2d = np.zeros(cfg.MODEL.IMAGE_SIZE)
        for j in range(c):
            heatmap += heatmaps[i][j]
        interior_landmarks = landmarks[i][0:8]
        iris_landmarks = landmarks[i][8:16]
        iris_centre = landmarks[i][-2]
        eyeball_centre = landmarks[i][-1]
        cv.polylines(frame_2d, [interior_landmarks], isClosed=True, color=(1, 1, 1), thickness=2)
        cv.polylines(frame_2d, [iris_landmarks], isClosed=True, color=(1, 1, 1), thickness=2)
        cv.line(frame_2d, tuple(eyeball_centre), tuple(iris_centre), color=(1, 1, 1), thickness=2)
        frame = heatmap + frame_2d
        valmax = np.max(frame)
        valmin = np.min(frame)
        frames[i] = frame/(valmax-valmin)
    frames = np.expand_dims(frames, axis=1)
    return torch.tensor(frames, dtype=torch.float32).cuda()


def get_gaze_frame_v2(heatmaps, landmarks):
    """
    Input: heatmaps, landmarks (torch.tensor in gpu) (n, c, h, w)
    Output: gaze_frame combined with heatmaps and landmarks (torch.tensor in gpu) (n, 1, h, w)
    """
    heatmaps = heatmaps.cpu().detach().numpy()
    landmarks = landmarks.cpu().detach().numpy()

    n, c, h, w = heatmaps.shape
    landmarks = np.floor((landmarks + 0.5), dtype=np.float)
    landmarks = np.int32(landmarks)
    frames = np.zeros((n, h, w))
    for i in range(n):
        heatmap = np.zeros(cfg.MODEL.IMAGE_SIZE)
        frame_2d = np.zeros(cfg.MODEL.IMAGE_SIZE)
        for j in range(c):
            heatmap += heatmaps[i][j]
        iris_centre = landmarks[i][-2]
        eyeball_centre = landmarks[i][-1]
        cv.line(frame_2d, tuple(eyeball_centre), tuple(iris_centre), color=(1, 1, 1), thickness=2)
        frame = heatmap + frame_2d
        valmax = np.max(frame)
        valmin = np.min(frame)
        frames[i] = frame/(valmax-valmin)
    frames = np.expand_dims(frames, axis=1)
    return torch.tensor(frames, dtype=torch.float32).cuda()


def get_gaze_frame_v3(heatmaps, landmarks):
    """
    Input: heatmaps, landmarks (torch.tensor in gpu) (n, c, h, w)
    Output: gaze_frame combined with heatmaps and landmarks (torch.tensor in gpu) (n, 1, h, w)
    """
    heatmaps = heatmaps.cpu().detach().numpy()
    landmarks = landmarks.cpu().detach().numpy()

    n, c, h, w = heatmaps.shape
    landmarks = np.floor((landmarks + 0.5), dtype=np.float)
    landmarks = np.int32(landmarks)
    frames = np.zeros((n, h, w))
    for i in range(n):
        frame_2d = np.zeros(cfg.MODEL.IMAGE_SIZE)
        interior_landmarks = landmarks[i][0:8]
        iris_landmarks = landmarks[i][8:16]
        iris_centre = landmarks[i][-2]
        eyeball_centre = landmarks[i][-1]
        cv.polylines(frame_2d, [interior_landmarks], isClosed=True, color=(1, 1, 1), thickness=2)
        cv.polylines(frame_2d, [iris_landmarks], isClosed=True, color=(1, 1, 1), thickness=2)
        cv.line(frame_2d, tuple(eyeball_centre), tuple(iris_centre), color=(1, 1, 1), thickness=2)
        frame = frame_2d
        valmax = np.max(frame)
        valmin = np.min(frame)
        frames[i] = frame/(valmax-valmin)
    frames = np.expand_dims(frames, axis=1)
    return torch.tensor(frames, dtype=torch.float32).cuda()


class FrameNet(nn.Module):
    def __init__(self, num_layers, high_resolution:bool = True):
        super(FrameNet, self).__init__()
        # Fist downsample.
        if high_resolution:
            self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        else:
            self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(1, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        # Flatten.
        self.flatten = nn.Flatten()
        # Input size 2160(Frame2d) + 1(Radius).
        # Output gaze (pitch, yaw).
        self.linear_regressor = self._make_layer(num_layers)

    def _make_layer(self, num_layers):
        linear_before = nn.Sequential(
            nn.Linear(2161, 100),
            nn.BatchNorm1d(100, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )
        linear = []
        for i in range(num_layers):
            linear.append(nn.Sequential(
                nn.Linear(100, 100),
                nn.BatchNorm1d(100, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True)
            ))
        linear_after = nn.Linear(100, 2)
        return nn.Sequential(
            linear_before,
            *linear,
            linear_after
        )

    def forward(self, x, radius):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        # (n, 2160) + (n, 1)
        x = self.flatten(x)
        x = torch.cat((x, radius), dim=-1)
        gaze = self.linear_regressor(x)

        return gaze


def get_frame_net(pretrained="", high_resolution:bool = True):
    if os.path.isfile(pretrained):
        logger.info('=> init frame_net weights from pretrained model')
        model = torch.load(pretrained)
    else:
        logger.info('=> init frame_net weights by default')
        model = FrameNet(num_layers=3, high_resolution=high_resolution)
        model = model.cuda()

    return model
