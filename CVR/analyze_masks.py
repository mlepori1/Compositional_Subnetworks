import os
import sys
# import yaml sys
import argparse
import copy
from CVR.models.resnet import L0Conv2d
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
                l0_components=["stage_4"],
                train_masks={"backbone": False, "mlp": False, "embedding": False},
                train_weights={"backbone": False, "mlp": False, "embedding": False},
                pretrained_weights={"backbone": "/gpfs/data/epavlick/mlepori/projects/Compositional_Subnetworks/Models/resnet50/Param_Sweep/Inside/L0_-03_stage4+mlp/backbone.pt",
                                    "mlp": "/gpfs/data/epavlick/mlepori/projects/Compositional_Subnetworks/Models/resnet50/Param_Sweep/Inside/L0_-03_stage4+mlp/mlp.pt",
                                    "embedding": False},
                eval_only=True,
                ablate_mask=None,
                l0_init=0,
                lamb=0.000000001)
                                
    cnn2 = modules.CNN(backbone="L0resnet50", 
                mlp_dim=128, 
                mlp_hidden_dim=256, 
                l0_components=["stage_4"],
                train_masks={"backbone": False, "mlp": False, "embedding": False},
                train_weights={"backbone": False, "mlp": False, "embedding": False},
                pretrained_weights={"backbone": "/gpfs/data/epavlick/mlepori/projects/Compositional_Subnetworks/Models/resnet50/Param_Sweep/Contact/L0_-05_stage4+mlp/backbone.pt",
                                    "mlp": "/gpfs/data/epavlick/mlepori/projects/Compositional_Subnetworks/Models/resnet50/Param_Sweep/Contact/L0_-05_stage4+mlp/mlp.pt",
                                    "embedding": False},
                eval_only=True,
                ablate_mask=None,
                l0_init=0,
                lamb=0.000000001)

    cnn1_l4 = cnn1.backbone.layer4.modules()
    cnn2_l4 = cnn2.backbone.layer4.modules()

    for layer_idx in len(cnn1_l4):
        layer1 = cnn1_l4[layer_idx]
        if isinstance(layer1, L0Conv2d):
            layer2 = cnn2_l4[layer_idx]

            mask1 = layer1.compute_mask()
            mask2 = layer2.compute_mask()

            print("Mask1 params: ", mask1.sum())
            print("Mask2 params: ", mask2.sum())
            print("Mask intesection: ", torch.logical_and(mask1, mask2).sum())

if __name__ == '__main__':
    print(os.getpid())

    cli_main()
