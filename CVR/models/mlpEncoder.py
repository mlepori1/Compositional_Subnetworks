import torch.nn as nn
#from ....L0_Masking.L0_Linear import L0UnstructuredLinear

#class L0MLP(nn.Module):
#    def __init__(self, mlp):
#        super(L0MLP, self).__init__()
#        self.model = nn.Sequential()
#        for layer in mlp.children():
#            if isinstance(layer, nn.Linear):
    #             self.model.append(L0UnstructuredLinear.from_module(layer))
    #         else:
    #             self.model.append(layer)

    #     self.embed_size = list(self.model.children())[-1].out_features # last layer is an L0 layer

    # def forward(self, input):
    #     return self.model(input)

    # def train(self, train_bool):
    #     for layer in self.modules():
    #         try:
    #             layer.train(train_bool)
    #         except:
    #             continue

#Linear network to prune after training
class MLP(nn.Module):
    def __init__(self, in_dim=128*128, dims=[2048, 2048, 1024, 1024, 1024, 1024]):
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
            nn.Linear(dims[4], dims[5])
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
