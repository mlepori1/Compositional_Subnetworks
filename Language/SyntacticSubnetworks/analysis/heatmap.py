import yaml
import os
import copy
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns

sns.set()
fig, ax = plt.subplots(figsize=(12, 5))

TASK_ID_2_STRING = {
    1: "(S) Subj.",
    2: "(S) Vb.",
    6: "(P) Subj.",
    7: "(P) Vb.",
    11: "(S) Pro.",
    12: "(S) Ant.",
    16: "(P) Pro",
    17: "(P) Ant.",
}

TASK_ID_2_COUNT = {
    1: 1,
    2: 1,
    6: 1,
    7: 1,
    11: 1,
    12: 1,
    16: 1,
    17: 1
}

data_dict = {
        "Model": [],
        "Mask": [],
        "Performance": []
    }

def extract_data(dir_path, hypoth_dict, failed_models):
    # First add singular task data to figure
    result_dirs = os.listdir(dir_path)

    data_dict = {
        "Model": [],
        "Mask": [],
        "Performance": []
    }
    # Model number determined by alph order of model name, totally arbitrary anyway
    result_dirs.sort()
    for idx, result_dir in enumerate(result_dirs):
        model_id = int(np.floor(idx/2)) # results are ordered according to model name, so every two results files indicate a new model
        if model_id in failed_models: 
            continue
        else:
            for mask_task in hypoth_dict:
                if "T" + str(mask_task) in result_dir:

                    sub_performance = [] # Difference between subnet performance on the two tasks 
                    abl_performance = [] # Difference between the ablated network on the two tasks

                    high_after_ablation = hypoth_dict[mask_task]["high_after_ablation"]
                    low_after_ablation = hypoth_dict[mask_task]["low_after_ablation"]

                    data = pd.read_csv(os.path.join(dir_path, result_dir, "results.csv"))

                    # Assert that all models ran 3 times
                    assert len(data["0_Model_ID"].unique()) == 3

                    # Model ID here corresponds to each run of the mask search algorithm
                    per_model_subnet_results = []
                    per_model_ablation_results = []

                    grouped_data = data.groupby("0_Model_ID")
                    for _, grp in grouped_data:

                        low_sub_acc = grp[(grp["0_ablation"].isna()) & (grp["1_task"] == int(high_after_ablation)) & (grp["0_train"] == 0)]["2_test_acc"].item()
                        high_sub_acc = grp[(grp["0_ablation"].isna()) & (grp["1_task"] == int(low_after_ablation)) & (grp["0_train"] == 0)]["2_test_acc"].item()

                        low_sub_abl_acc = grp[(grp["0_ablation"] == "zero") & (grp["1_task"] == int(low_after_ablation)) & (grp["0_train"] == 0)]["2_test_acc"].item()
                        high_sub_abl_acc = grp[(grp["0_ablation"] == "zero") & (grp["1_task"] == int(high_after_ablation)) & (grp["0_train"] == 0)]["2_test_acc"].item()
                        
                        per_model_subnet_results.append(np.max([high_sub_acc, .25]) - np.max([low_sub_acc, .25]))
                        per_model_ablation_results.append(np.max([low_sub_abl_acc, .25]) - np.max([high_sub_abl_acc, .25]))

                    print(mask_task)
                    data_dict["Performance"] += per_model_subnet_results + per_model_ablation_results
                    data_dict["Model"] += [TASK_ID_2_STRING[int(mask_task)] + " " + str(TASK_ID_2_COUNT[int(mask_task)])] * 6
                    TASK_ID_2_COUNT[int(mask_task)] += 1
                    data_dict["Mask"] += ["Subnetwork " + str(i) for i in range(1, 4)] + ["Ablation " + str(i) for i in range(1, 4)]
    return data_dict
            
stream = open("../configs/Graph_Configs/heatmaps.yaml", 'r')
config = yaml.load(stream, Loader=yaml.FullLoader)
stream.close()

# First add SV singular task data to figure
if "failed_models_sv_sing" in config.keys():
    failed_models = config["failed_models_sv_sing"]
else: 
    failed_models = []
data = extract_data(config["sv_sing_dir"], config["hypothesized_performance_sv_sing"], failed_models)
data_dict["Model"] += data["Model"]
data_dict["Performance"] += data["Performance"]
data_dict["Mask"] += data["Mask"]

# add SV plural task data to figure
pl_tasks = list(config["hypothesized_performance_sv_pl"].keys())
pl_tasks = [int(t) for t in pl_tasks]
if "failed_models_sv_pl" in config.keys():
    failed_models = config["failed_models_sv_pl"]
else: 
    failed_models = []
data = extract_data(config["sv_pl_dir"], config["hypothesized_performance_sv_pl"], failed_models)
data_dict["Model"] += data["Model"]
data_dict["Performance"] += data["Performance"]
data_dict["Mask"] += data["Mask"]

# add Anaphora Singular task data to figure
if "failed_models_an_sing" in config.keys():
    failed_models = config["failed_models_an_sing"]
else: 
    failed_models = []
data = extract_data(config["an_sing_dir"], config["hypothesized_performance_an_sing"], failed_models)
data_dict["Model"] += data["Model"]
data_dict["Performance"] += data["Performance"]
data_dict["Mask"] += data["Mask"]

# add Anaphora Plural task data to figure
if "failed_models_an_pl" in config.keys():
    failed_models = config["failed_models_an_pl"]
else: 
    failed_models = []
data = extract_data(config["an_pl_dir"], config["hypothesized_performance_an_pl"], failed_models)
data_dict["Model"] += data["Model"]
data_dict["Performance"] += data["Performance"]
data_dict["Mask"] += data["Mask"]

print(data_dict)
df = pd.DataFrame.from_dict(data_dict)
print(df)

graph_path = os.path.join(config["outdir"], "heatmap_random.png")

piv = df.pivot('Mask', 'Model',
               'Performance')

piv = df.pivot(index='Mask',
               columns='Model',
               values='Performance'
               )

hmap = sns.heatmap(
    annot=False,
    cmap="Spectral",
    cbar_kws={'label': 'Performance'},
    data=piv,
    vmin=-0.75,
    vmax=0.75,
    square=True
)
hmap.set_title('Subnetwork Performance: Randomly Initialized Models (Language)', fontsize=16)
hmap.set_xticklabels(hmap.get_xticklabels(), rotation = 70, fontsize = 10)
hmap.set_xlabel(hmap.get_xlabel(), fontsize=14)
hmap.set_ylabel(hmap.get_ylabel(), fontsize=14)

plt.savefig(graph_path)