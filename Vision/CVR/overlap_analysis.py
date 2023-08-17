import os
import sys
import shutil
# import yaml sys
import argparse
import copy
import torch
import numpy as np
import pandas as pd
import copy

import pytorch_lightning as pl

from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks import TQDMProgressBar
from main import MetricsCallback, TemperatureCallback
import matplotlib.pyplot as plt

from models.model import Model
from utils import parse_args, save_config, find_best_epoch, process_results

argv = sys.argv[1:]

config_prefix = "configs/Inside_Count/Resnet50/Overlap_Analysis/Mask_Configs/883cd15e-cf44-43bd-8fe9-56a91090aef7_T"

task2intersection = {}

plot_data = {}
suffix2task = {"110": "Inside", "111": "Number"}

for task_suffix in ["110", "111"]:
    config = config_prefix + task_suffix + ".yaml"
    parser = argparse.ArgumentParser()
    args = parse_args(parser, argv, config=config) # Here is where variables from the config file override command line args

    # trainer args
    parser = pl.Trainer.add_argparse_args(parser)

    # model args
    model_type = Model
    parser = model_type.add_model_specific_args(parser)

    models = []
    for pt_weights in args.pretrained_weights_list:
        print(pt_weights)
        args.pretrained_weights = pt_weights
        models.append(model_type(**args.__dict__))

    # First perform intersection analysis

    # Gather up Mask Weight keys
    model = models[0]
    mask_weight_keys = []
    for layer in model.state_dict():
        if "mask_weight" in layer:
            mask_weight_keys.append(layer)

    iou_dict = {}
    intersection_dict = {}
    for layer in mask_weight_keys:
        model0 = models[0]
        mask0 = model0.state_dict()[layer] > 0

        model1 = models[1]
        mask1 = model1.state_dict()[layer] > 0

        model2 = models[2]
        mask2 = model2.state_dict()[layer] > 0

        # Compute the Intersection
        intersection = torch.logical_and(mask0, mask1)
        intersection = torch.logical_and(intersection, mask2)

        # Compute the Union
        union = torch.logical_or(mask0, mask1)
        union = torch.logical_or(intersection, mask2)

        iou_dict[layer] = torch.sum(intersection)/torch.sum(union)
        intersection_dict[layer] = intersection

    print("Within Task IOU: " + task_suffix)
    plot_data[suffix2task[task_suffix]] = iou_dict
    for k, v in iou_dict.items():
        print(f"{k}: {v}")

    task2intersection[task_suffix] = intersection_dict

overlapping_layers = list(set(task2intersection["110"].keys()).intersection(task2intersection["111"].keys()))
print(overlapping_layers)
overlapping_layers.sort()

iou_dict = {}
for layer in overlapping_layers:
    mask0 = task2intersection["110"][layer]
    mask1 = task2intersection["111"][layer]

    # Compute the Intersection
    intersection = torch.logical_and(mask0, mask1)

    # Compute the Union
    union = torch.logical_or(mask0, mask1)

    iou_dict[layer] = torch.sum(intersection)/torch.sum(union)

print("Between Task IOU")
plot_data["Intersection"] = iou_dict

for k, v in iou_dict.items():
    print(f"{k}: {v}")

plt.figure(figsize=(10, 10))
plt.title("Overlap Analysis")
plt.ylim(0, 1.05)
plt.ylabel("IoU")
plt.xlabel("Layers")
for k, v in plot_data.items():
    label = k
    layers = []
    ious = []
    for layer, iou in v.items():
        layers.append(layer[:-12])
        ious.append(iou)

    plt.plot(list(range(len(layers))), ious, label=label)

plt.xticks(list(range(len(layers))), layers, rotation=90)
plt.legend()
plt.tight_layout()
plt.savefig("overlap.pdf", format="pdf")