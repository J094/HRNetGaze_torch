# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Jun Guo (jun.guo.chn@outlook.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

import torch
import torch.nn as nn

from src.configs import cfg


BN_MOMENTUM = 0.1
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(True)

    def _check_branches(self, num_branches, blocks, num_blocks,
                        num_inchannels, num_channels):
        # num_blocks, num_channels, num_inchannels should be defined for each branch.
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         stride=1):
        # Check if this branch need downsample, only new branch need.
        # (If stride = 1 for new branch, then just add a parallel branch with same size)
        downsample = None
        if stride != 1 or \
           self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index] * block.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(
                    num_channels[branch_index] * block.expansion,
                    momentum=BN_MOMENTUM
                ),
            )

        layers = []
        layers.append(
            block(
                self.num_inchannels[branch_index],
                num_channels[branch_index],
                stride,
                downsample
            )
        )
        # If a new branch did downsample, make num_inchannels = num_channels * expansion.
        # Also ensure this branch will not do downsample in the further steps.
        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(
                block(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index]
                )
            )
        # Add a layer list into a nn.Sequential module.
        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []
        # Make branches according to the num_branches of every stage.
        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels)
            )
        # These branches are connected complex, so use nn.ModuleList to store.
        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        # If there is only one branch in this stage, no need to do fuse.
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        # fuse_layers manage branches.
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            # fuse_layer manege layers to branch i.
            fuse_layer = []
            # Remember from branch j to branch i, there is only one layer appended into fuse_layer.
            for j in range(num_branches):
                # Do upsample from every lower branch j to higher branch i.
                if j > i:
                    fuse_layer.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_inchannels[j],
                                num_inchannels[i],
                                1, 1, 0, bias=False
                            ),
                            nn.BatchNorm2d(num_inchannels[i]),
                            nn.Upsample(scale_factor=2**(j-i), mode='nearest')
                        )
                    )
                # In the same branch, do nothing.
                elif j == i:
                    fuse_layer.append(None)
                # Do downsample from every higher branch j to lower branch i.
                else:
                    conv3x3s = []
                    for k in range(i-j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_outchannels_conv3x3,
                                        3, 2, 1, bias=False
                                    ),
                                    nn.BatchNorm2d(num_outchannels_conv3x3)
                                )
                            )
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_outchannels_conv3x3,
                                        3, 2, 1, bias=False
                                    ),
                                    nn.BatchNorm2d(num_outchannels_conv3x3),
                                    nn.ReLU(True)
                                )
                            )
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))
        # Add layer list into nn.ModuleList.
        return nn.ModuleList(fuse_layers)

    # Return num_inchannels for creating next stage.
    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        # If only have one branch, go though it.
        if self.num_branches == 1:
            return [self.branches[0](x[0])]
        # If have multiple branches, first go though each branch separately.
        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])
        # Use list to manage outputs of each branch.
        # (Also inputs for next loop step)
        x_fuse = []
        # Do a loop if have multiple fuse_layers.
        for i in range(len(self.fuse_layers)):
            # First add layer output from first branch to y.
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            # Add layer output from the rest branches to y.
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))
        # x_fuse contains outputs from all branches.
        return x_fuse


blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}


class GazeHighResolutionNet(nn.Module):

    def __init__(self, cfg, **kwargs):
        self.inplanes = 64
        # Load config informations for HRNet.
        extra = cfg['MODEL']['EXTRA']
        super(GazeHighResolutionNet, self).__init__()

        # stage 1
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(Bottleneck, 64, 2)

        # stage 2
        self.stage2_cfg = extra['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']  # branch1: 32, branch2: 64
        block = blocks_dict[self.stage2_cfg['BLOCK']]  # block: BasicBlock
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition1 = self._make_transition_layer([256], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels)

        # stage 3
        self.stage3_cfg = extra['STAGE3']
        num_channels = self.stage3_cfg['NUM_CHANNELS']  # branch1: 32, branch2: 64, branch3: 128
        block = blocks_dict[self.stage3_cfg['BLOCK']]  # block: BasicBlock
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition2 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels)

        # stage 4
        self.stage4_cfg = extra['STAGE4']
        num_channels = self.stage4_cfg['NUM_CHANNELS']  # branch1: 32, branch2: 64, branch3: 128, branch4: 256
        block = blocks_dict[self.stage4_cfg['BLOCK']]  # block: BasicBlock
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition3 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels, multi_scale_output=False)

        self.final_layer = nn.Conv2d(
            in_channels=pre_stage_channels[0],
            out_channels=cfg['MODEL']['NUM_LDMKS'],
            kernel_size=extra['FINAL_CONV_KERNEL'],
            stride=1,
            padding=1 if extra['FINAL_CONV_KERNEL'] == 3 else 0
        )
        # Calculate landmarks from heatmaps.
        self.callandmarks = CalLandmarks(cfg['MODEL']['NUM_LDMKS'], cfg['MODEL']['HEATMAPS_SCALE'])
        # 4-layer Linears as radius_regressor.
        self.radius_regressor = RadiusRegressor(cfg['MODEL']['NUM_LDMKS'])
        # self.pretrained_layers = extra['PRETRAINED_LAYERS']

    def _make_transition_layer(
            self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                # Branch i has been created, so just check if num_channels match.
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_channels_pre_layer[i],
                                num_channels_cur_layer[i],
                                3, 1, 1, bias=False
                            ),
                            nn.BatchNorm2d(num_channels_cur_layer[i]),
                            nn.ReLU(inplace=True)
                        )
                    )
                # If match, then append None.
                else:
                    transition_layers.append(None)
            else:
                # Branch i is a new branch, which need to be created. (Downsample)
                conv3x3s = []
                # According to how far the new branch to the last old branch in pre_stage,
                # make the correct num of Downsample layers.
                for j in range(i+1-num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    # Ensure outchannels for last new branch.
                    outchannels = num_channels_cur_layer[i] \
                        if j == i-num_branches_pre else inchannels
                    conv3x3s.append(
                        nn.Sequential(
                            nn.Conv2d(
                                inchannels, outchannels, 3, 2, 1, bias=False
                            ),
                            nn.BatchNorm2d(outchannels),
                            nn.ReLU(inplace=True)
                        )
                    )
                transition_layers.append(nn.Sequential(*conv3x3s))
        # transition_layers work in different branches, so use nn.ModuleList to manage.
        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, planes, blocks, stride=1):
        # Check if this step needs downsample.
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes, planes * block.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        # If need more blocks, do loop.
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels,
                    multi_scale_output=True):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            modules.append(
                HighResolutionModule(
                    num_branches,
                    block,
                    num_blocks,
                    num_inchannels,
                    num_channels,
                    fuse_method,
                    reset_multi_scale_output
                )
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):
        # Go through stage1.
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list = []
        # Go through transition1.
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        # Go through stage2.
        y_list = self.stage2(x_list)

        x_list = []
        # Go through transition2.
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        # Go through stage3.
        y_list = self.stage3(x_list)

        # If input is a high_resolution image.
        if cfg['MODEL']['HIGH_RESOLUTION']:
            x_list = []
            # Go through transition3.
            for i in range(self.stage4_cfg['NUM_BRANCHES']):
                if self.transition3[i] is not None:
                    x_list.append(self.transition3[i](y_list[-1]))
                else:
                    x_list.append(y_list[i])
            # Go through stage4.
            y_list = self.stage4(x_list)
        
        # Final layer to predict 18 heatmaps for 18 landmarks.
        heatmaps = self.final_layer(y_list[0])
        # Calculate coordinaten of 18 landmarks from heatmaps. (No trainable!)
        ldmks = self.callandmarks(heatmaps)
        # Linear module to do radius regression.
        radius = self.radius_regressor(ldmks)

        return heatmaps, ldmks, radius

    def init_weights(self):
        logger.info('=> init gaze_net weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class CalLandmarks(nn.Module):
    def __init__(self, num_landmarks, heatmaps_scale):
        super(CalLandmarks, self).__init__()
        self.num_landmarks = num_landmarks
        self.heatmaps_scale = heatmaps_scale
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        _, _, h, w = x.shape
        h = int(h / self.heatmaps_scale)
        w = int(w / self.heatmaps_scale)
        # Assume normalized coordinate [0, 1] for numeric stability
        ref_ys, ref_xs = torch.meshgrid(torch.linspace(0, 1.0, steps=h),
                                        torch.linspace(0, 1.0, steps=w))
        ref_xs = torch.reshape(ref_xs, (-1, h*w)).cuda()
        ref_ys = torch.reshape(ref_ys, (-1, h*w)).cuda()
        # Assuming NHWC, for PyTorch it's NCHW, don't need transpose
        beta = 1e2
        # Transpose x from NHWC to NCHW
        # x = torch.transpose(x, 1, 3)
        # x = torch.transpose(x, 2, 3)
        x = torch.reshape(x, (-1, self.num_landmarks, h*w))
        x = self.softmax(beta*x)
        lmrk_xs = torch.sum(ref_xs * x, dim=2)
        lmrk_ys = torch.sum(ref_ys * x, dim=2)
        # Return to actual coordinates ranges
        # The label heatmaps + 0.5px, so we only need + 0.5 here
        return torch.stack([
            lmrk_xs * (w - 1.0) + 0.5,
            lmrk_ys * (h - 1.0) + 0.5
        ], dim=2)  # N x 18 x 2


class RadiusRegressor(nn.Module):
    def __init__(self, num_landmarks):
        super(RadiusRegressor, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_radius = self._make_layer(num_landmarks)
    
    def _make_layer(self, num_landmarks):
        linear_before = nn.Sequential(
            nn.Linear(2*num_landmarks, 100),
            nn.BatchNorm1d(100, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )
        linear = []
        for i in range(3):
            linear.append(nn.Sequential(
                nn.Linear(100, 100),
                nn.BatchNorm1d(100, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True)
            ))
        linear_after = nn.Linear(100, 1)
        return nn.Sequential(
            linear_before,
            *linear,
            linear_after
            )
        
    def forward(self, x):
        x = self.flatten(x)
        x = self.linear_radius(x)
        return x


def get_gaze_net(cfg, pretrained: str, **kwargs):
    if os.path.isfile(pretrained):
        logger.info('=> init gaze_net weights from pretrained model')
        model = torch.load(pretrained)
    else:
        model = GazeHighResolutionNet(cfg, **kwargs)
        model.init_weights()
        model = model.cuda()

    return model
