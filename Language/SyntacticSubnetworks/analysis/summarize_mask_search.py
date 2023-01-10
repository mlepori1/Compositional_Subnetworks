import yaml
import os
import copy
import argparse
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, help='Config file containing the summarization arguments')
args = parser.parse_args()

stream = open(args.config, 'r')
config = yaml.load(stream, Loader=yaml.FullLoader)
stream.close()

summary_data = []

## 1) Iterate through all base models, Mask Training tasks
base_models = os.listdir(config["base_dir"])
for base_model in base_models:
    mask_training_tasks = os.listdir(os.path.join(config["base_dir"], base_model))
    for mask_task in mask_training_tasks:
        mask_configurations = os.listdir(os.path.join(config["base_dir"], base_model, mask_task))

        high_after_ablation = config["hypothesized_performance"][mask_task]["high_after_ablation"]
        low_after_ablation = config["hypothesized_performance"][mask_task]["low_after_ablation"]

        ## 2) Concatenate all masked models together within a base model + mask training task
        data = []
        for mask_configuration in mask_configurations:
            try:
                data.append(pd.read_csv(os.path.join(config["base_dir"], base_model, mask_task, mask_configuration, "results.csv")))
            except:
                print(base_model)
                print(mask_configuration)
        data = pd.concat(data)

        ## 3) Iterate through masked models and compute subroutine quality
        quality_data = []
        grouped_data = data.groupby("0_Model_ID")
        for model_id, grp in grouped_data:
            # First, filter based on train task subroutine performance on validation set
            subnetwork_acc = grp[(grp["0_ablation"].isna()) & (grp["1_task"] == int(mask_task)) & (grp["0_train"] == 0)]["2_test_acc"].item()
            if subnetwork_acc < .9: 
                quality_score = 0.0
            else:
                # Next, compute ablation quality using zero-ablation
                high_acc = grp[(grp["0_ablation"] == "zero") & (grp["1_task"] == int(high_after_ablation)) & (grp["0_train"] == 0)]["2_test_acc"].item()
                low_acc = grp[(grp["0_ablation"] == "zero") & (grp["1_task"] == int(low_after_ablation)) & (grp["0_train"] == 0)]["2_test_acc"].item()
                quality_score = np.max([high_acc, .25]) - np.max([low_acc, .25]) # Max is 0.75, Min is 0.0

            grp["Quality Score"] = len(grp) * [quality_score]
            grp["Base Model"] = len(grp) * [base_model]
            grp["0_Model_ID"] = len(grp) * [model_id]

            quality_data.append(grp)

        quality_data = pd.concat(quality_data)

        ## 4) Find argmax of subroutine quality, append to a DF
        max_quality = np.max(quality_data["Quality Score"])
        best_masked_model = quality_data[quality_data["Quality Score"] == max_quality]["0_Model_ID"].iloc[0]
        summary_data.append(quality_data[quality_data["0_Model_ID"] == best_masked_model])

## 5) Save summary DF
summary_data = pd.concat(summary_data)
summary_data.to_csv(os.path.join(config["base_dir"], "summary.csv"))