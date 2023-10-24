import yaml
import os
import copy
import argparse
import numpy as np
import pandas as pd

BASE_DIR = "../../Results/Count_Contact/Resnet50/Pruning_Parameters/"

summary_data = []

## 1) Iterate through all base models, Mask Training tasks
base_models = os.listdir(BASE_DIR)
for base_model in base_models:
    mask_training_tasks = os.listdir(os.path.join(BASE_DIR, base_model))
    for mask_task in mask_training_tasks:
        mask_configurations = os.listdir(os.path.join(BASE_DIR, base_model, mask_task))

        ## 2) Concatenate all masked models together within a base model + mask training task
        data = []
        for mask_configuration in mask_configurations:
            try:
                data.append(pd.read_csv(os.path.join(BASE_DIR, base_model, mask_task, mask_configuration, "results.csv")))
            except:
                print(base_model)
                print(mask_configuration)
        data = pd.concat(data)

        # Get just relevant rows
        data = data[(data["0_ablation"].isna()) & (data["1_task"] == int(mask_task)) & (data["0_train"] == 0)]

        ## Get the configuration with best test acc
        high_acc = np.max(data["2_test_acc"])
        best_masked_model = data[data["2_test_acc"] == high_acc].iloc[0].to_frame().T
        best_masked_model["Base Model"] = [base_model]
        summary_data.append(best_masked_model)

## 5) Save summary DF
summary_data = pd.concat(summary_data)
summary_data.to_csv(os.path.join(BASE_DIR, "summary.csv"))