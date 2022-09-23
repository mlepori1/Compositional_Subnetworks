import torch.nn as nn
import torch
import torch.nn.functional as F

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
        mask_init_value=0.0,
        temp=1
    ) -> None:
        """Initialize a L0UstructuredLinear module.

        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.mask_init_value = mask_init_value
        self.weight = nn.Parameter(torch.zeros(in_features, out_features))  # type: ignore
        self.temp = temp
        self.init_mask()

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))  # type: ignore
        else:
            self.register_parameter("bias", None)  # type: ignore

        self.compiled_weight = None
        self.reset_parameters()

    def init_mask(self):
        self.mask_weight = nn.Parameter(torch.zeros(self.in_features, self.out_features))
        nn.init.constant_(self.mask_weight, self.mask_initial_value)

    def compute_mask(self):
        scaling = 1. / nn.sigmoid(self.mask_initial_value)
        if not self.training: mask = (self.mask_weight > 0).float()
        else: mask = F.sigmoid(self.temp * self.mask_weight)
        return scaling * mask      

    #@todo still gotta implement the forward pass, and the from_module, prune, and calc_l0 methods 
    def train(self, train_bool):
        self.training = train_bool

    @classmethod
    def from_module(
        self,
        module: nn.Linear,
        init_mean: float = 0.5,
        init_std: float = 0.01,
        keep_weights: bool = True,
    ) -> "L0UnstructedLinear":
        """Construct from a pretrained nn.Linear module.
        IMPORTANT: the weights are conserved, but can be reinitialized
        with `keep_weights = False`.
        Parameters
        ----------
        module: nn.Linear
            A ``nn.Linear`` module.
        init_mean : float, optional
            Initialization value for hard concrete parameter,
            by default 0.5.,
        init_std: float, optional
            Used to initialize the hard concrete parameters,
            by default 0.01.
        Returns
        -------
        L0UnstructedLinear
            The input module with a hardconcrete mask introduced.
        """
        in_features = module.in_features
        out_features = module.out_features
        bias = module.bias is not None
        new_module = self(in_features, out_features, bias, init_mean, init_std)

        if keep_weights:
            new_module.weight.data = module.weight.data.transpose(0, 1).clone()
            if bias:
                new_module.bias.data = module.bias.data.clone()

        return new_module

    def reset_parameters(self):
        """Reset network parameters."""
        self.mask.reset_parameters() # Keep mask reset
        init.kaiming_uniform_(self.weight, a=math.sqrt(5)) # Update Linear reset to match torch 1.12 https://pytorch.org/docs/stable/_modules/torch/nn/modules/linear.html#Linear
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def num_prunable_parameters(self) -> int:
        """Get number of prunable parameters"""
        return self.in_features * self.out_features

    def num_parameters(self, train=True) -> torch.Tensor:
        """Get number of parameters."""
        if self.training:
            # Get the expected number of parameters
            n_active = self.mask.l0_norm()
        elif self.compiled_weight is not None:
            n_active = torch.sum(self.compiled_weight.reshape(-1) != 0)
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
        if self.training:
            # First reset the compiled weights
            self.compiled_weight = None

            # Sample, and compile dynamically
            mask = self.mask()
            compiled_weight = self.weight * mask
            U = data.matmul(compiled_weight)

        else:
            if self.compiled_weight is None:
                mask = self.mask()
                compiled_weight = self.weight * mask
                self.compiled_weight = compiled_weight

            U = data.matmul(self.compiled_weight)  # type: ignore

        return U if self.bias is None else U + self.bias

    def extra_repr(self) -> str:
        s = "in_features={in_features}, out_features={out_features}"
        s += ", bias={}".format(str(self.bias is not None))
        return s.format(**self.__dict__)

    def __repr__(self) -> str:
        return "{}({})".format(self.__class__.__name__, self.extra_repr())

class L0MLP(nn.Module):
    def __init__(self, mlp):
        super(L0MLP, self).__init__()
        self.model = nn.Sequential()
        for layer in mlp.children():
            if isinstance(layer, nn.Linear):
                self.model.append(L0UnstructuredLinear.from_module(layer))
            else:
                self.model.append(layer)

        self.embed_size = list(self.model.children())[-1].out_features # last layer is an L0 layer

    def forward(self, input):
        return self.model(input)

    def train(self, train_bool):
        for layer in self.modules():
            try:
                layer.train(train_bool)
            except:
                continue

#Linear network to prune after training
class MLP(nn.Module):
    def __init__(self, in_dim=128*128, dims=[2048, 2048, 1024, 1024, 1024, 1024, 512]):
        super(MLP, self).__init__()
        self.embed_size = dims[-1]
        self.in_dim = in_dim 
        self.model = nn.Sequential(
            nn.Linear(in_dim, dims[0]),
            nn.ReLU(),
            nn.Linear(dims[0], dims[1]),
            nn.ReLU(),
            nn.Linear(dims[1], dims[2]),
            nn.ReLU(),
            nn.Linear(dims[2], dims[3]),
            nn.ReLU(),
            nn.Linear(dims[3], dims[4]),
            nn.ReLU(),
            nn.Linear(dims[4], dims[5]),
            nn.ReLU(),
            nn.Linear(dims[5], dims[6])
        )

    def forward(self, input):
        input = input[:, 0, :, :] # Input is grayscaled, so just grab one dimension
        input = input.reshape(-1, self.in_dim)
        return self.model(input)

    def train(self, train_bool):
        for layer in self.modules():
            try:
                layer.train(train_bool)
            except:
                continue
