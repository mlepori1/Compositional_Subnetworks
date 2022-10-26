# From the Continuous Sparsification Repo

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np
import torch.nn.init as init
import math
import functools
from .resnet import L0Conv2d

class ResBlock(nn.Module):
    def __init__(self, Conv, in_channels, out_channels, stride=1, downsample=None, batch_norm=False):
        super(ResBlock, self).__init__()
        self.conv_a = Conv(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.conv_b = Conv(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if batch_norm:
            self.bn_a = nn.BatchNorm2d(out_channels)
            self.bn_b = nn.BatchNorm2d(out_channels)
        else:
            self.bn_a = nn.Identity()
            self.bn_b = nn.Identity()
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv_a(x)
        out = self.bn_a(out)
        out = F.relu(out, inplace=True)
        out = self.conv_b(out)
        out = self.bn_b(out)
        if self.downsample is not None: residual = self.downsample(x)
        return F.relu(residual + out, inplace=True)
    
class ResStage(nn.Module):
    def __init__(self, Conv, in_channels, out_channels, stride=1, batch_norm=False):
        super(ResStage, self).__init__()
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Conv2d(in_channels, out_channels, 1, 2, 0, bias=False)
            
        self.block1 = ResBlock(Conv, in_channels, out_channels, stride, downsample, batch_norm=batch_norm)
        self.block2 = ResBlock(Conv, out_channels, out_channels, batch_norm=batch_norm)
        self.block3 = ResBlock(Conv, out_channels, out_channels, batch_norm=batch_norm)

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        return out

class ResNet18(nn.Module):
    def __init__(self, isL0=False, mask_init_value=0., embed_dim=10, batch_norm=True, ablate_mask=None, l0_stages=None):
        super(ResNet18, self).__init__()

        self.isL0 = isL0
        self.bn = batch_norm
        self.ablate_mask = ablate_mask # Used during testing to see performance when found mask is removed

        if l0_stages != None:
            self.l0_stages = l0_stages
        else:
            self.l0_stages = ["first", "stage_1", "stage_2", "stage_3"]

        L0_Conv = functools.partial(L0Conv2d, l0=True, mask_init_value=mask_init_value, ablate_mask=self.ablate_mask)
        Conv = functools.partial(L0Conv2d, l0=False)

        if isL0 and "first" in self.l0_stages:
            print("first")
            self.conv0 = L0Conv2d(3, 16, 3, 1, 1, l0=False)
        else:
            self.conv0 = Conv(3, 16, 3, 1, 1)

        if self.bn:
            self.bn0 = nn.BatchNorm2d(16)
        else:
            self.bn0 = nn.Identity()

        if isL0 and "stage_1" in self.l0_stages:
            print("stage_1")
            self.stage1 = ResStage(L0_Conv, 16, 16, stride=1, batch_norm=self.bn)
        else:
            self.stage1 = ResStage(Conv, 16, 16, stride=1, batch_norm=self.bn)

        if isL0 and "stage_2" in self.l0_stages:
            print("stage_2")
            self.stage2 = ResStage(L0_Conv, 16, 32, stride=2, batch_norm=self.bn)
        else:
            self.stage2 = ResStage(Conv, 16, 32, stride=2, batch_norm=self.bn)

        if isL0 and "stage_3" in self.l0_stages:
            print("stage_3")
            self.stage3 = ResStage(L0_Conv, 32, 64, stride=2, batch_norm=self.bn)
        else:
            self.stage3 = ResStage(Conv, 32, 64, stride=2, batch_norm=self.bn)

        self.avgpool = nn.AvgPool2d(8)
        self.embed_dim=embed_dim
        self.mask_modules = [m for m in self.modules() if type(m) == L0Conv2d]
        self.temp = 1.

    def get_temp(self):
        return self.temp

    def set_temp(self, temp):
        self.temp = temp
        for layer in self.modules():
            if type(layer) == L0Conv2d:
                layer.temp = temp
                
    def forward(self, x):
        out = F.relu(self.bn0(self.conv0(x)), inplace=True)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.avgpool(out)
        out = out.view(x.size(0), -1)
        return out
