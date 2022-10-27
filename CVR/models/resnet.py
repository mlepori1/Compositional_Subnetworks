from functools import partial
from typing import Any, Callable, List, Optional, Type, Union

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
import torch.nn.init as init
import math

import functools

__all__ = [
    "ResNet",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "resnext50_32x4d",
    "resnext101_32x8d",
    "resnext101_64x4d",
    "wide_resnet50_2",
    "wide_resnet101_2",
]

class L0Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1, dilation=1, groups=1, bias=False, l0=False, mask_init_value=0., temp: float = 1., ablate_mask=None):
        super(L0Conv2d, self).__init__()
        self.l0 = l0
        self.mask_init_value = mask_init_value
        
        self.in_channels = in_channels
        self.out_channels = out_channels    
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.dilation = _pair(dilation)
        self.groups = groups
        self.temp = temp
        self.ablate_mask=ablate_mask

        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        # Create a random tensor to reinit ablated parameters
        if self.ablate_mask == "random":
            self.random_weight = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
            init.kaiming_uniform_(self.random_weight, a=math.sqrt(5))
            self.random_weight.requires_grad=False

        
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
        if (self.ablate_mask == None) and (not self.training or self.mask_weight.requires_grad == False): 
            mask = (self.mask_weight > 0).float() # Hard cutoff once frozen or testing
        elif (self.ablate_mask != None) and (not self.training or self.mask_weight.requires_grad == False): 
            mask = (self.mask_weight <= 0).float() # Used for subnetwork ablation
        else:
            mask = F.sigmoid(self.temp * self.mask_weight)
        return mask 

    def train(self, train_bool):
        self.training = train_bool         
        
    def forward(self, x):
        if self.l0:
            self.mask = self.compute_mask()
            if self.ablate_mask == "random":
                masked_weight = self.weight * self.mask # This will give you the inverse weights, 0's for ablated weights
                masked_weight += (~self.mask.bool()).float() * self.random_weight# Invert the mask to target the remaining weights, make them random
            else:
                masked_weight = self.weight * self.mask
        else:
            masked_weight = self.weight
        out = F.conv2d(x, masked_weight, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)        
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


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1, conv_fn=functools.partial(L0Conv2d, l0=False)) -> L0Conv2d:
    """3x3 convolution with padding"""
    return conv_fn(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1, conv_fn=functools.partial(L0Conv2d, l0=False)) -> L0Conv2d:
    """1x1 convolution"""
    return conv_fn(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        conv_fn = functools.partial(L0Conv2d, l0=False)
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride, conv_fn=conv_fn)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, conv_fn=conv_fn)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        conv_fn = functools.partial(L0Conv2d, l0=False)
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width, conv_fn=conv_fn)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation, conv_fn=conv_fn)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion, conv_fn=conv_fn)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        print(x.shape)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        print(out.shape)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        print(out.shape)

        out = self.conv3(out)
        out = self.bn3(out)
        print(out.shape)

        if self.downsample is not None:
            identity = self.downsample(x)

        print(identity.shape)
        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        isL0: bool = False,
        mask_init_value: float = 0.,
        embed_dim:int = 10,
        ablate_mask:str = None,
        l0_stages:list = None
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        # Added parameters
        self.isL0 = isL0
        self.ablate_mask = ablate_mask # Used during testing to see performance when found mask is removed
        self.embed_dim = embed_dim
        self.temp = 1.

        if l0_stages != None:
            self.l0_stages = l0_stages
        else:
            self.l0_stages = ["first", "stage_1", "stage_2", "stage_3", "stage_4"]

        # Set up two layer constructors
        L0_Conv = functools.partial(L0Conv2d, l0=True, mask_init_value=mask_init_value, ablate_mask=self.ablate_mask)
        Conv = functools.partial(L0Conv2d, l0=False)

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group

        # Set up the network layers
        if self.isL0 and "first" in self.l0_stages:
            print("first")
            self.conv1 = L0_Conv(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        else:
            self.conv1 = Conv(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)

        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        if self.isL0 and "stage_1" in self.l0_stages:
            print("stage1")
            self.layer1 = self._make_layer(block, 64, layers[0], conv_fn=L0_Conv)
        else:
            self.layer1 = self._make_layer(block, 64, layers[0], conv_fn=Conv)

        if self.isL0 and "stage_2" in self.l0_stages:
            print("stage2")
            self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0], conv_fn=L0_Conv)
        else:
            self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0], conv_fn=Conv)

        if self.isL0 and "stage_3" in self.l0_stages:
            print("stage3")
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1], conv_fn=L0_Conv)
        else:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1], conv_fn=Conv)
        
        if self.isL0 and "stage_4" in self.l0_stages:
            print("stage4")
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2], conv_fn=L0_Conv)
        else:
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2], conv_fn=Conv)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, L0Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
        conv_fn:Type[Union[nn.Conv2d, L0Conv2d]] = functools.partial(L0Conv2d, l0=False)
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride, conv_fn=conv_fn),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer, conv_fn=conv_fn
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                    conv_fn=conv_fn
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

    def get_temp(self):
        return self.temp

    def set_temp(self, temp):
        self.temp = temp
        for layer in self.modules():
            if type(layer) == L0Conv2d:
                layer.temp = temp


def _resnet(
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    **kwargs: Any,
) -> ResNet:

    model = ResNet(block, layers, **kwargs)
    return model


def resnet18(**kwargs: Any) -> ResNet:
    """ResNet-18 from `Deep Residual Learning for Image Recognition <https://arxiv.org/pdf/1512.03385.pdf>`__.

    Args:
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.ResNet18_Weights
        :members:
    """
    return _resnet(BasicBlock, [2, 2, 2, 2], **kwargs)

def resnet34(**kwargs: Any) -> ResNet:
    """ResNet-34 from `Deep Residual Learning for Image Recognition <https://arxiv.org/pdf/1512.03385.pdf>`__.

    Args:
        weights (:class:`~torchvision.models.ResNet34_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.ResNet34_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.ResNet34_Weights
        :members:
    """
    return _resnet(BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet50(**kwargs: Any) -> ResNet:
    """ResNet-50 from `Deep Residual Learning for Image Recognition <https://arxiv.org/pdf/1512.03385.pdf>`__.

    .. note::
       The bottleneck of TorchVision places the stride for downsampling to the second 3x3
       convolution while the original paper places it to the first 1x1 convolution.
       This variant improves the accuracy and is known as `ResNet V1.5
       <https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch>`_.

    Args:
        weights (:class:`~torchvision.models.ResNet50_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.ResNet50_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.ResNet50_Weights
        :members:
    """

    return _resnet(Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet101(**kwargs: Any) -> ResNet:
    """ResNet-101 from `Deep Residual Learning for Image Recognition <https://arxiv.org/pdf/1512.03385.pdf>`__.

    .. note::
       The bottleneck of TorchVision places the stride for downsampling to the second 3x3
       convolution while the original paper places it to the first 1x1 convolution.
       This variant improves the accuracy and is known as `ResNet V1.5
       <https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch>`_.

    Args:
        weights (:class:`~torchvision.models.ResNet101_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.ResNet101_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.ResNet101_Weights
        :members:
    """

    return _resnet(Bottleneck, [3, 4, 23, 3], **kwargs)


def resnet152(**kwargs: Any) -> ResNet:
    """ResNet-152 from `Deep Residual Learning for Image Recognition <https://arxiv.org/pdf/1512.03385.pdf>`__.

    .. note::
       The bottleneck of TorchVision places the stride for downsampling to the second 3x3
       convolution while the original paper places it to the first 1x1 convolution.
       This variant improves the accuracy and is known as `ResNet V1.5
       <https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch>`_.

    Args:
        weights (:class:`~torchvision.models.ResNet152_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.ResNet152_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.ResNet152_Weights
        :members:
    """

    return _resnet(Bottleneck, [3, 8, 36, 3], **kwargs)

def resnext50_32x4d(
    **kwargs: Any
) -> ResNet:
    """ResNeXt-50 32x4d model from
    `Aggregated Residual Transformation for Deep Neural Networks <https://arxiv.org/abs/1611.05431>`_.

    Args:
        weights (:class:`~torchvision.models.ResNeXt50_32X4D_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.ResNext50_32X4D_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.
    .. autoclass:: torchvision.models.ResNeXt50_32X4D_Weights
        :members:
    """

    kwargs["groups"] = 32
    kwargs["width_per_group"] = 4
    return _resnet(Bottleneck, [3, 4, 6, 3], **kwargs)

def resnext101_32x8d(
    **kwargs: Any
) -> ResNet:
    """ResNeXt-101 32x8d model from
    `Aggregated Residual Transformation for Deep Neural Networks <https://arxiv.org/abs/1611.05431>`_.

    Args:
        weights (:class:`~torchvision.models.ResNeXt101_32X8D_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.ResNeXt101_32X8D_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.
    .. autoclass:: torchvision.models.ResNeXt101_32X8D_Weights
        :members:
    """

    kwargs["groups"] = 32
    kwargs["width_per_group"] =  8
    return _resnet(Bottleneck, [3, 4, 23, 3], **kwargs)

def resnext101_64x4d(
    **kwargs: Any
) -> ResNet:
    """ResNeXt-101 64x4d model from
    `Aggregated Residual Transformation for Deep Neural Networks <https://arxiv.org/abs/1611.05431>`_.

    Args:
        weights (:class:`~torchvision.models.ResNeXt101_64X4D_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.ResNeXt101_64X4D_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.
    .. autoclass:: torchvision.models.ResNeXt101_64X4D_Weights
        :members:
    """

    kwargs["groups"] = 64
    kwargs["width_per_group"] = 4
    return _resnet(Bottleneck, [3, 4, 23, 3], **kwargs)


def wide_resnet50_2(
    **kwargs: Any
) -> ResNet:
    """Wide ResNet-50-2 model from
    `Wide Residual Networks <https://arxiv.org/abs/1605.07146>`_.

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        weights (:class:`~torchvision.models.Wide_ResNet50_2_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.Wide_ResNet50_2_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.
    .. autoclass:: torchvision.models.Wide_ResNet50_2_Weights
        :members:
    """

    kwargs["width_per_group"] = 64 * 2
    return _resnet(Bottleneck, [3, 4, 6, 3], **kwargs)


def wide_resnet101_2(
    **kwargs: Any
) -> ResNet:
    """Wide ResNet-101-2 model from
    `Wide Residual Networks <https://arxiv.org/abs/1605.07146>`_.

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-101 has 2048-512-2048
    channels, and in Wide ResNet-101-2 has 2048-1024-2048.

    Args:
        weights (:class:`~torchvision.models.Wide_ResNet101_2_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.Wide_ResNet101_2_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.
    .. autoclass:: torchvision.models.Wide_ResNet101_2_Weights
        :members:
    """
    kwargs["width_per_group"] = 64 * 2
    return _resnet(Bottleneck, [3, 4, 23, 3], **kwargs)
