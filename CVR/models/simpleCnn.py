from resnet18 import L0Conv2d
import torch.nn as nn
import functools
import torch.nn.functional as F

class simpleCNN(nn.Module):
    def __init__(self, isL0=False, mask_init_value=0., embed_dim=10):
        super(simpleCNN, self).__init__()

        self.isL0 = isL0

        if isL0:
            Conv = functools.partial(L0Conv2d, l0=True, mask_init_value=mask_init_value)
        else:
            Conv = functools.partial(L0Conv2d, l0=False)

        self.conv0 = Conv(3, 16, 3, 1, 1)
        self.conv1 = Conv(16, 32, 3, 2, 1)
        self.conv2 = Conv(32, 64, 3, 2, 1)

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
        x = F.relu(self.conv0(x), inplace=True)
        x = F.relu(self.conv1(x), inplace=True)
        out = F.relu(self.conv2(x), inplace=True)
        out = out.view(x.size(0), -1)
        return out
