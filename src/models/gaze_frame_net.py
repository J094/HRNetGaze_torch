import logging

import torch
import torch.nn as nn
import cv2 as cv
import numpy as np

from configs import cfg
from models.gaze_hrnet import BasicBlock, Bottleneck


BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


def get_gaze_frame(heatmaps, landmarks):
    """
    Input: heatmaps, landmarks (torch.tensor in gpu)
    Output: gaze_frame combined with heatmaps and landmarks (torch.tensor in gpu)
    """
    heatmaps = heatmaps.cpu().detatch().numpy()
    landmarks = landmarks.cpu().detatch().numpy()

    n, c, h, w = heatmaps.shape
    landmarks = np.floor((landmarks + 0.5), dtype=np.int32)
    frames = np.zeroos((n, h, w))
    for i in range(n):
        heatmap = np.zeros(cfg.MODEL.IMAGE_SIZE)
        frame_2d = np.zeros(cfg.MODEL.IMAGE_SIZE)
        for j in range(c):
            heatmap += heatmaps[n][c]
        interior_landmarks = landmarks[i][0:8]
        iris_landmarks = landmarks[i][8:16]
        iris_centre = landmarks[i][-2]
        eyeball_centre = landmarks[i][-1]
        cv.polylines(frame_2d, [interior_landmarks], isClosed=True, color=(1, 1, 1), thickness=2)
        cv.polylines(frame_2d, [iris_landmarks], isClosed=True, color=(1, 1, 1), thickness=2)
        cv.line(frame_2d, tuple(eyeball_centre), tuple(iris_centre), color=(1, 1, 1), thickness=2)
        frames[i] = heatmap + frame_2d
    return torch.tensor(frames).cuda()


class FrameNet(nn.Module):
    def __init__(self):
        super(FrameNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        pass
