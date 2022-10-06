from .resnet18 import L0Conv2d
import torch.nn as nn
import functools
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self, isL0=False, mask_init_value=0., embed_dim=10):
        super(LeNet, self).__init__()

        self.isL0 = isL0

        if isL0:
            Conv = functools.partial(L0Conv2d, l0=True, mask_init_value=mask_init_value)
        else:
            Conv = functools.partial(L0Conv2d, l0=False)

        self.conv0 = Conv(3, 6, 5, padding=0, stride=1)
        self.conv1 = Conv(6, 16, 5, padding=0, stride=1)
        self.conv2 = Conv(16, 120, 5, padding=0, stride=1)
        self.tanh = nn.Tanh()
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
                print(layer.temp) # for debug
                
    def forward(self, x):
        x = self.conv1(x)
        x = self.tanh(x)
        x = self.avgpool(x)
        x = self.conv2(x)
        x = self.tanh(x)
        x = self.avgpool(x)
        x = self.conv3(x)
        x = self.tanh(x)
        
        out = x.reshape(x.shape[0], -1)
        return out
