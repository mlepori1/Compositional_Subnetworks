import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.nn.init as init
import math
import functools

class L0Linear(nn.Module):
    """The hard concrete equivalent of ``nn.Linear``.
        Pruning is unstructured, with weights masked at
        the individual parameter level, not neuron level
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        mask_init_value: float = 0.0,
        temp: float = 1.,
        l0: bool = False,
        ablate_mask: str = None
    ) -> None:
        """Initialize a L0UstructuredLinear module.

        """
        super().__init__()

        self.l0 = l0
        self.in_features = in_features
        self.out_features = out_features
        self.mask_init_value = mask_init_value
        self.weight = nn.Parameter(torch.zeros(out_features, in_features))  # type: ignore
        self.temp = temp
        self.ablate_mask = ablate_mask

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))  # type: ignore
        else:
            self.register_parameter("bias", None)  # type: ignore

        self.reset_parameters()

        # Create a random tensor to reinit ablated parameters
        if self.ablate_mask == "random":
            self.random_weight = nn.Parameter(torch.zeros(out_features, in_features))
            init.kaiming_uniform_(self.random_weight, a=math.sqrt(5))
            self.random_weight.requires_grad=False

    def init_mask(self):
        self.mask_weight = nn.Parameter(torch.zeros(self.out_features, self.in_features))
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

    def reset_parameters(self):
        """Reset network parameters."""
        if self.l0:
            self.init_mask()

        init.kaiming_uniform_(self.weight, a=math.sqrt(5)) # Update Linear reset to match torch 1.12 https://pytorch.org/docs/stable/_modules/torch/nn/modules/linear.html#Linear
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)


    def forward(self, data: torch.Tensor, **kwargs) -> torch.Tensor:  # type: ignore
        """Perform the forward pass.
        Parameters
        ----------
        data : torch.Tensor
            N-dimensional tensor, with last dimension `in_features`
        Returns
        -------
        torch.Tensor
            N-dimensional tensor, with last dimension `out_features`
        """
        if self.l0:
            self.mask = self.compute_mask()
            if self.ablate_mask == "random":
                masked_weight = self.weight * self.mask # This will give you the inverse weights, 0's for ablated weights
                masked_weight += (~self.mask.bool()).float() * self.random_weight# Invert the mask to target the 0'd weights, make them random
            else:
                masked_weight = self.weight * self.mask
        else:
            masked_weight = self.weight

        out = F.linear(data, masked_weight, self.bias)
        return out

    def extra_repr(self) -> str:
        s = "in_features={in_features}, out_features={out_features}"
        s += ", bias={}".format(str(self.bias is not None))
        return s.format(**self.__dict__)

    def __repr__(self) -> str:
        return "{}({})".format(self.__class__.__name__, self.extra_repr())

class MLP(nn.Module):
    def __init__(self, 
        in_dim, 
        hidden_dim, 
        out_dim, 
        isL0 = False,
        mask_init_value = 0, 
        ablate_mask=None):
        # MLP is either a regular MLP (such as the one defined below) 
        # or another unstructured L0 MLP. 
        # Ablate mask is used during testing to see how performance varies without 
        # the found subnetwork
        super(MLP, self).__init__()

        self.isL0 = isL0
        self.ablate_mask = ablate_mask # Used during testing to see performance when found mask is removed
        self.temp = 1.

        if self.isL0:
            linear = functools.partial(L0Linear, l0=True, mask_init_value=mask_init_value, ablate_mask=ablate_mask)
        else:
            linear = functools.partial(L0Linear, l0=False)

        if hidden_dim != 0:
            self.model = nn.Sequential(
                linear(in_dim, hidden_dim),
                nn.ReLU(),
                linear(hidden_dim, out_dim)
            )
        else:
            self.model = nn.Sequential(
                linear(in_dim, out_dim),
            )
        self.embed_dim = out_dim
    
    def forward(self, input):
        return self.model(input)

    def get_temp(self):
        return self.temp

    def set_temp(self, temp):
        self.temp = temp
        for layer in self.model.children():
            if isinstance(layer, L0Linear):
                layer.temp = temp


