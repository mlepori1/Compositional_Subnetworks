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


    for name, layer1 in cnn1.named_parameters():

        if "mask_weight" in name:
            for name2, layer2 in cnn2.named_parameters():
                if name2==name:
                    mask1 = layer1 > 0
                    mask2 = layer2 > 0

                    print("Mask1 params: ", mask1.sum())
                    print("Mask2 params: ", mask2.sum())
                    print("Mask intesection: ", torch.logical_and(mask1, mask2).sum())
                    print("Size of tensor: ", mask1.reshape(-1).size())
                    intersection_mask = torch.logical_and(mask1, mask2).float()
                    layer1.data = intersection_mask         
    
    
    for name, layer1 in cnn1.named_parameters():
        if "mask_weight" in name:
            print(layer1)
            print(layer1.sum())

    os.makedirs("/gpfs/data/epavlick/mlepori/projects/Compositional_Subnetworks/Models/resnet50/Param_Sweep/Inside/L0_-03_stage4+mlp_intersect/", exist_ok=True)

    torch.save(cnn1.backbone.state_dict(), os.path.join("/gpfs/data/epavlick/mlepori/projects/Compositional_Subnetworks/Models/resnet50/Param_Sweep/Inside/L0_-03_stage4+mlp_intersect/", 'backbone.pt'))
    torch.save(cnn1.mlp.state_dict(), os.path.join("/gpfs/data/epavlick/mlepori/projects/Compositional_Subnetworks/Models/resnet50/Param_Sweep/Inside/L0_-03_stage4+mlp_intersect/", 'mlp.pt'))
if __name__ == '__main__':
    print(os.getpid())

    cli_main()