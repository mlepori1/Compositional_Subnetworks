# From the Continuous Sparsification Repo

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np
import torch.nn.init as init
import math
import functools


class L0Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=1, stride=1, bias=False, l0=False, mask_init_value=0., temp: float = 1., inverse_mask=False):
        super(L0Conv2d, self).__init__()
        self.l0 = l0
        self.mask_init_value = mask_init_value
        
        self.in_channels = in_channels
        self.out_channels = out_channels    
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.temp = temp
        self.inverse_mask=inverse_mask

        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        
    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(k), 1/sqrt(k)), where k = weight.size(1) * prod(*kernel_size)
        # For more details see: https://github.com/pytorch/pytorch/issues/15314#issuecomment-477448573
        if self.l0:
            self.init_mask()

        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(self.bias, -bound, bound)
                
    def init_mask(self):
        self.mask_weight = nn.Parameter(torch.Tensor(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size))
        nn.init.constant_(self.mask_weight, self.mask_init_value)

    def compute_mask(self):
        if (not self.inverse_mask) and (not self.training or self.mask_weight.requires_grad == False): 
            mask = (self.mask_weight > 0).float() # Hard cutoff once frozen or testing
        elif (self.inverse_mask) and (not self.training or self.mask_weight.requires_grad == False): mask = (self.mask_weight <= 0).float() # Used for subnetwork ablation
        else:

        # FOR TESTING ONLY, REMOVE DISCRETE MASK
            mask = F.sigmoid(self.temp * self.mask_weight)
        return mask 

    def train(self, train_bool):
        self.training = train_bool         
        
    def forward(self, x):
        if self.l0:
            self.mask = self.compute_mask()
            masked_weight = self.weight * self.mask
        else:
            masked_weight = self.weight
        out = F.conv2d(x, masked_weight, stride=self.stride, padding=self.padding)        
        return out

    @classmethod
    def from_module(
        self,
        module,
        mask_init_value: float = 0.0,
        keep_weights: bool = True,
    ) -> "L0Conv2d":
        """Construct from a pretrained conv2d module.
        IMPORTANT: the weights are conserved, but can be reinitialized
        with `keep_weights = False`.
        Parameters
        ----------
        module: Conv2d, L0Conv2d
            A nn.Conv2d or L0Conv2d
        mask_init_value : float, optional
            Initialization value for the sigmoid .
        Returns
        -------
        L0Conv2d
            The input module with a CS mask introduced.
        """
        in_channels = module.in_channels
        out_channels = module.out_channels
        kernel_size = module.kernel_size
        padding = module.padding
        stride = module.stride        

        bias = module.bias is not None
        if hasattr(module, "mask_weight"):
            mask = module.mask_weight
        else:
            mask = None
        new_module = self(in_channels, out_channels, kernel_size, bias=bias, padding=padding, stride=stride, mask_init_value=mask_init_value)

        if keep_weights:
            new_module.weight.data = module.weight.data.clone()
            if bias:
                new_module.bias.data = module.bias.data.clone()
            if mask:
                new_module.mask_weight.data = module.mask_weight.data.clone()

        return new_module

    def extra_repr(self):
        return '{}, {}, kernel_size={}, stride={}, padding={}'.format(
            self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding)

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

class ResNet(nn.Module):
    def __init__(self, isL0=False, mask_init_value=0., embed_dim=10, batch_norm=True, ablate_mask=False):
        super(ResNet, self).__init__()

        self.isL0 = isL0
        self.bn = batch_norm
        self.ablate_mask = ablate_mask # Used during testing to see performance when found mask is removed

        if isL0:
            Conv = functools.partial(L0Conv2d, l0=True, mask_init_value=mask_init_value, inverse_mask=ablate_mask)
        else:
            Conv = functools.partial(L0Conv2d, l0=False)

        self.conv0 = Conv(3, 16, 3, 1, 1)
        if self.bn:
            self.bn0 = nn.BatchNorm2d(16)
        else:
            self.bn0 = nn.Identity()

        self.stage1 = ResStage(Conv, 16, 16, stride=1, batch_norm=self.bn)
        self.stage2 = ResStage(Conv, 16, 32, stride=2, batch_norm=self.bn)
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
