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

class VGG11(nn.Module):
    def __init__(self, isL0=False, mask_init_value=0., embed_dim=10, ablate_mask=None):
        super(VGG11, self).__init__()

        """
        Implements VGG11 convolutional backbone, to be fed into decision MLP
        """

        self.isL0 = isL0
        self.ablate_mask = ablate_mask # Used during testing to see performance when found mask is removed

        if isL0:
            Conv = functools.partial(L0Conv2d, l0=True, mask_init_value=mask_init_value, ablate_mask=ablate_mask)
        else:
            Conv = functools.partial(L0Conv2d, l0=False)

        self.conv_layers = nn.Sequential(
            Conv(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            Conv(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            Conv(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            Conv(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

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
                print(layer.temp) # for debug
                
    def forward(self, x):
        out = self.conv_layers(x)
        out = out.reshape(out.size(0), -1)
        return out
