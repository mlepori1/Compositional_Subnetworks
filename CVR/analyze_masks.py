import os
import sys
# import yaml sys
import argparse
import copy
from models.resnet import L0Conv2d
from models.decisionMLP import L0UnstructuredLinear
import torch
import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl

import modules


def cli_main():
    cnn1 = modules.CNN(backbone="L0resnet50", 
                mlp_dim=128, 
                mlp_hidden_dim=256, 
                l0_stages=["stage_4"],
                l0_components={"backbone": True, "mlp": True, "embedding": False},
                train_masks={"backbone": False, "mlp": False, "embedding": False},
                train_weights={"backbone": False, "mlp": False, "embedding": False},
                pretrained_weights={"backbone": "/gpfs/data/epavlick/mlepori/projects/Compositional_Subnetworks/Models/resnet50/Param_Sweep/Inside/L0_-03_stage4+mlp/backbone.pt",
                                    "mlp": "/gpfs/data/epavlick/mlepori/projects/Compositional_Subnetworks/Models/resnet50/Param_Sweep/Inside/L0_-03_stage4+mlp/mlp.pt",
                                    "embedding": False},
                eval_only=True,
                ablate_mask=None,
                l0_init=0,
                l0_lambda=0.000000001)
                                
    cnn2 = modules.CNN(backbone="L0resnet50", 
                mlp_dim=128, 
                mlp_hidden_dim=256, 
                l0_stages=["stage_4"],
                l0_components={"backbone": True, "mlp": True, "embedding": False},
                train_masks={"backbone": False, "mlp": False, "embedding": False},
                train_weights={"backbone": False, "mlp": False, "embedding": False},
                pretrained_weights={"backbone": "/gpfs/data/epavlick/mlepori/projects/Compositional_Subnetworks/Models/resnet50/Param_Sweep/Inside/L0_-03_stage4+mlp_1/backbone.pt",
                                    "mlp": "/gpfs/data/epavlick/mlepori/projects/Compositional_Subnetworks/Models/resnet50/Param_Sweep/Inside/L0_-03_stage4+mlp_1/mlp.pt",
                                    "embedding": False},
                eval_only=True,
                ablate_mask=None,
                l0_init=0,
                l0_lambda=0.000000001)
    for name, w in cnn1.backbone.layer4.named_parameters():
        print(name)
    print("CNN L4")
    cnn1_l4 = list(cnn1.backbone.layer4.modules())
    cnn2_l4 = list(cnn2.backbone.layer4.modules())
    for layer_idx in range(len(cnn1_l4)):
        layer1 = cnn1_l4[layer_idx]
        if isinstance(layer1, L0Conv2d):
            layer2 = cnn2_l4[layer_idx]

            mask1 = layer1.compute_mask()
            mask2 = layer2.compute_mask()
            print(layer1)
            print("Mask1 params: ", mask1.sum())
            print("Mask2 params: ", mask2.sum())
            print("Mask intesection: ", torch.logical_and(mask1, mask2).sum())
            print("Size of tensor: ", mask1.reshape(-1).size())
            
    
    print("MLP")
    cnn1_mlp = list(cnn1.mlp.model.modules())
    cnn2_mlp = list(cnn2.mlp.model.modules())
    for layer_idx in range(len(cnn1_mlp)):
        layer1 = cnn1_mlp[layer_idx]
        if isinstance(layer1, L0UnstructuredLinear):
            layer2 = cnn2_mlp[layer_idx]

            mask1 = layer1.compute_mask()
            mask2 = layer2.compute_mask()

            print("Mask1 params: ", mask1.sum())
            print("Mask2 params: ", mask2.sum())
            print("Mask intesection: ", torch.logical_and(mask1, mask2).sum())
            print("Size of tensor: ", mask1.reshape(-1).size())
            
if __name__ == '__main__':
    print(os.getpid())

    cli_main()
