import yaml
import os
import copy
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

TASK_ID_2_STRING = {
    104: "Inside",
    105: "Contact",
    107: "+/- Inside",
    108: "+/- Contact",
    110: "Inside",
    111: "Number",
    113: "+/- Number",
    114: "+/- Inside",
    116: "Contact",
    117: "Number",
    119: "+/- Number",
    120: "+/- Contact",
}

TASK_ID_2_OVERALL_TASK = {
    104: "Inside-Contact",
    105: "Inside-Contact",
    110: "Inside-Number",
    111: "Inside-Number",
    116: "Number-Contact",
    117: "Number-Contact"
}

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, help='Config file containing the summarization arguments')
args = parser.parse_args()

stream = open(args.config, 'r')
config = yaml.load(stream, Loader=yaml.FullLoader)
stream.close()

# Create outdir
os.makedirs(config["outdir"], exist_ok=True)

#Delete existing figures
if os.path.exists(os.path.join(config["base_dir"], "disag_difference.png")):
  os.remove(os.path.join(config["base_dir"], "disag_difference.png"))
if os.path.exists(os.path.join(config["base_dir"], "summary_difference.png")):
  os.remove(os.path.join(config["base_dir"], "summary_difference.png"))
if os.path.exists(os.path.join(config["base_dir"], "task_difference.png")):
  os.remove(os.path.join(config["base_dir"], "task_difference.png"))
  
if "failed_models" in config.keys():
    failed_models = config["failed_models"]
else: 
    failed_models = []

cols = []
train_tasks = list(config["hypothesized_performance"].keys())
train_tasks = [int(t) for t in train_tasks]
train_tasks.sort()

for k in train_tasks:
    cols.append(TASK_ID_2_STRING[int(k)]) # First column for lower train task
    fig_title = TASK_ID_2_OVERALL_TASK[int(k)]

# Task-Agg'd figure
t_fig, t_axs = plt.subplots(nrows=1, ncols=2, figsize=(5.5, 5))
for ax in t_axs: 
    ax.set_ylim(-1, 1)
    ax.set_xlim(-.1, .35)
    ax.axhline(y=0.0, color='gray', linestyle='--')

t_fig.suptitle(fig_title, fontsize=15)

for ax, col in zip(t_axs, cols):
    ax.set_ylabel(col, rotation=90, fontsize=13)

# Summary figure
sub_color = "b"
abl_color = "r"

model_markers = ["o", "^", "s"]

result_dirs = os.listdir(config["base_dir"])

def add_data_to_figure(axes, dir_path, hypoth_dict, edgecolor, alpha, tasks_list, x_locs, failed_models):

    data_for_significance = {
        "Base_Model": [],
        "Ablation": [],
        "Task": [],
        "Performance": []
    }
    # First add singular task data to figure
    result_dirs = os.listdir(dir_path)

    # Model number determined by alph order of model name, totally arbitrary anyway
    result_dirs.sort()
    for idx, result_dir in enumerate(result_dirs):
        print("\n\n")
        print("BASE MODEL: " + str(int(np.floor(idx/2))))
        print(result_dir)
        for mask_task in hypoth_dict:
            if "T" + str(mask_task) in result_dir:
                print(TASK_ID_2_STRING[int(mask_task)])

                sub_performance = [] # Difference between subnet performance on the two tasks 
                abl_performance = [] # Difference between the ablated network on the two tasks

                high_after_ablation = hypoth_dict[mask_task]["high_after_ablation"]
                low_after_ablation = hypoth_dict[mask_task]["low_after_ablation"]

                data = pd.read_csv(os.path.join(dir_path, result_dir, "results.csv"))

                # Assert that all models ran 3 times
                assert len(data["0_Model_ID"].unique()) == 3

                # Model ID here corresponds to each run of the mask search algorithm
                grouped_data = data.groupby("0_Model_ID")
                for model_id, grp in grouped_data:
                    print(model_id)
                    low_sub_acc = grp[(grp["0_ablation"].isna()) & (grp["1_task"] == int(high_after_ablation)) & (grp["0_train"] == 0)]["2_test_acc"].item()
                    high_sub_acc = grp[(grp["0_ablation"].isna()) & (grp["1_task"] == int(low_after_ablation)) & (grp["0_train"] == 0)]["2_test_acc"].item()
                    low_sub_abl_acc = grp[(grp["0_ablation"] == "zero") & (grp["1_task"] == int(low_after_ablation)) & (grp["0_train"] == 0)]["2_test_acc"].item()
                    high_sub_abl_acc = grp[(grp["0_ablation"] == "zero") & (grp["1_task"] == int(high_after_ablation)) & (grp["0_train"] == 0)]["2_test_acc"].item()

                    sub_performance.append(np.max([high_sub_acc, .25]) - np.max([low_sub_acc, .25]))
                    abl_performance.append(np.max([low_sub_abl_acc, .25]) - np.max([high_sub_abl_acc, .25]))

                # Add data
                # Establish one train task per col    
                if int(mask_task) == np.min(tasks_list): # first col for lower train task
                    task_col = 0
                else:
                    task_col = 1

                model_id = int(np.floor(idx/2)) # results are ordered according to model name, so every two results files indicate a new model

                # Task aggregated Plot
                ax = t_axs[task_col]

                x = x_locs + (model_id * .05)# the label locations

                if model_id == 1:
                    ax.set_xticks(x, ["Subnetwork", "Ablation"], fontsize=13)

                if model_id in failed_models:
                    sub = ax.scatter(np.repeat([x[0]], len(sub_performance)), sub_performance, c="gray", alpha=alpha, marker=model_markers[model_id], edgecolors=edgecolor)
                    abl = ax.scatter(np.repeat([x[1]],  len(abl_performance)), abl_performance, c="gray", alpha=alpha, marker=model_markers[model_id], edgecolors=edgecolor)
                else:
                    sub = ax.scatter(np.repeat([x[0]], len(sub_performance)), sub_performance, c=sub_color, alpha=alpha, marker=model_markers[model_id], edgecolors=edgecolor)
                    abl = ax.scatter(np.repeat([x[1]],  len(abl_performance)), abl_performance, c=abl_color, alpha=alpha, marker=model_markers[model_id], edgecolors=edgecolor)

                    data_for_significance["Base_Model"] += [model_id] * len(sub_performance)
                    data_for_significance["Ablation"] += [0] * len(sub_performance)
                    data_for_significance["Task"] += [cols[task_col]] * len(sub_performance)
                    data_for_significance["Performance"] += sub_performance

                    data_for_significance["Base_Model"] += [model_id] * len(abl_performance)
                    data_for_significance["Ablation"] += [1] * len(abl_performance)
                    data_for_significance["Task"] += [cols[task_col]] * len(abl_performance)
                    data_for_significance["Performance"] += abl_performance

    return data_for_significance

data_for_significance = add_data_to_figure(t_axs, config["base_dir"], config["hypothesized_performance"], "face", 1.0, train_tasks, np.arange(2)/4  - .05, failed_models)

# Save the figure and show
plt.tight_layout()

# Create summary legend
legend_elements = [
    t_axs[0].scatter([], [], color='black', marker=model_markers[0], label='Seed 1'),
    t_axs[0].scatter([], [], color='black', marker=model_markers[1], label='Seed 2'),
    t_axs[0].scatter([], [], color='black', marker=model_markers[2], label='Seed 3')
]

#for ax in t_axs: ax.legend(handles=legend_elements)

t_fig.tight_layout()
graph_path = os.path.join(config["outdir"], "task_difference.png")
t_fig.savefig(graph_path)

pd.DataFrame.from_dict(data_for_significance).to_csv(os.path.join(config["outdir"], "analysis_data.csv"), index=False)
