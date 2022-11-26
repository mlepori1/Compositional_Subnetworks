import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.nn.init as init
import math

class L0UnstructuredLinear(nn.Module):
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
        ablate_mask: str = None
    ) -> None:
        """Initialize a L0UstructuredLinear module.

        """
        super().__init__()

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

        # Create a random tensor to reinit ablated parameters
        if self.ablate_mask == "random":
            self.random_weight = nn.Parameter(torch.zeros(out_features, in_features))
            init.kaiming_uniform_(self.random_weight, a=math.sqrt(5))
            self.random_weight.requires_grad=False

        self.reset_parameters()

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

    @classmethod
    def from_module(
        self,
        module: nn.Linear,
        mask_init_value: float = 0.0,
        keep_weights: bool = True,
        ablate_mask: str = None
    ) -> "L0UnstructedLinear":
        """Construct from a pretrained nn.Linear module.
        IMPORTANT: the weights are conserved, but can be reinitialized
        with `keep_weights = False`.
        Parameters
        ----------
        module: nn.Linear, L0UnstructuredLinear
            A ``nn.Linear`` module, or another L0Unstructured linear model.
        mask_init_value : float, optional
            Initialization value for the sigmoid .
        Returns
        -------
        L0UnstructedLinear
            The input module with a CS mask introduced.
        """
        in_features = module.in_features
        out_features = module.out_features
        bias = module.bias is not None
        if hasattr(module, "mask_weight"):
            mask = module.mask_weight
        else:
            mask = None
        new_module = self(in_features, out_features, bias, mask_init_value, ablate_mask=ablate_mask)

        if keep_weights:
            new_module.weight.data = module.weight.data.clone()
            if bias:
                new_module.bias.data = module.bias.data.clone()
            if mask:
                new_module.mask_weight.data = module.mask_weight.data.clone()

        return new_module

    def reset_parameters(self):
        """Reset network parameters."""
        self.init_mask() # Keep mask reset
        init.kaiming_uniform_(self.weight, a=math.sqrt(5)) # Update Linear reset to match torch 1.12 https://pytorch.org/docs/stable/_modules/torch/nn/modules/linear.html#Linear
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def num_prunable_parameters(self) -> int:
        """Get number of prunable parameters"""
        return self.weight.size()

    def num_parameters(self) -> torch.Tensor:
        """Get number of active test parameters."""
        train = self.training
        self.training = False
        n_active = torch.sum(self.compute_mask() > 0)
        self.training = train
        return n_active

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
        self.mask = self.compute_mask()
        if self.ablate_mask == "random":
            masked_weight = self.weight * self.mask # This will give you the inverse weights, 0's for ablated weights
            masked_weight += (~self.mask.bool()).float() * self.random_weight# Invert the mask to target the remaining weights, make them random
        else:
            masked_weight = self.weight * self.mask

        out = F.linear(data, masked_weight, self.bias)
        return out

    def extra_repr(self) -> str:
        s = "in_features={in_features}, out_features={out_features}"
        s += ", bias={}".format(str(self.bias is not None))
        return s.format(**self.__dict__)

    def __repr__(self) -> str:
        return "{}({})".format(self.__class__.__name__, self.extra_repr())

class L0MLP(nn.Module):
    def __init__(self, mlp, mask_init_value, ablate_mask=None):
        # MLP is either a regular MLP (such as the one defined below) 
        # or another unstructured L0 MLP. 
        # Ablate mask is used during testing to see how performance varies without 
        # the found subnetwork
        super(L0MLP, self).__init__()
        self.model = []
        for layer in mlp.model.children():
            if isinstance(layer, nn.Linear):
                self.model.append(L0UnstructuredLinear.from_module(layer, mask_init_value=mask_init_value, ablate_mask=ablate_mask))
            else:
                self.model.append(layer)
        self.embed_size = self.model[-1].out_features # last layer is an L0 layer
        self.in_dim = self.model[0].in_features
        self.model = nn.Sequential(*self.model)
    
    def forward(self, input):
        return self.model(input)

    def get_temp(self):
        for layer in self.model.children():
            if isinstance(layer, L0UnstructuredLinear):
                return layer.temp

    def set_temp(self, temp):
        for layer in self.model.children():
            if isinstance(layer, L0UnstructuredLinear):
                layer.temp = temp
 
    def calibration_mode(self):
        for layer in self.model.modules():
            if hasattr(layer, "weight"):
                layer.weight.requires_grad = False
            if hasattr(layer, "bias"):
                layer.bias.requires_grad = False
            if hasattr(layer, "mask_weight"):
                layer.mask_weight.requires_grad = False

#Linear network to prune after training
class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(MLP, self).__init__()
        self.in_dim = in_dim
        if hidden_dim != 0:
            self.model = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, out_dim)
            )
        else:
            self.model = nn.Sequential(
                nn.Linear(in_dim, out_dim),
            )

    def forward(self, input):
        return self.model(input)

    def train(self, train_bool):
        for layer in self.modules():
            try:
                layer.train(train_bool)
            except:
                continue
