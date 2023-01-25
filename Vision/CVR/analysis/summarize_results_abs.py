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

    110: "Inside",
    111: "Number",

    116: "Contact",
    117: "Number",

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
  
if "failed_models" in config.keys():
    failed_models = config["failed_models"]
else: 
    failed_models = []

cols = []
train_tasks = list(config["hypothesized_performance"].keys())
train_tasks = [int(t) for t in train_tasks]
train_tasks.sort()

for k in train_tasks:
    cols.append(TASK_ID_2_STRING[int(k)]) # First col for lower train task
    fig_title = TASK_ID_2_OVERALL_TASK[int(k)]

# Sub Figure
s_fig, s_axs = plt.subplots(nrows=1, ncols=2, figsize=(5.5, 5))
for ax in s_axs: 
    ax.set_ylim(0, 1.1)
    ax.set_xlim(-.1, .35)
s_fig.suptitle(config["model"] + " Subnetworks: " + fig_title, fontsize=15)

for ax, col in zip(s_axs, cols):
    ax.set_ylabel(col, rotation=90, fontsize=13)

# Abl Figure
a_fig, a_axs = plt.subplots(nrows=1, ncols=2, figsize=(5.5, 5))
for ax in a_axs: 
    ax.set_ylim(0, 1.1)
    ax.set_xlim(-.1, .35)
a_fig.suptitle(config["model"] + " Ablated Networks: " + fig_title, fontsize=15)

for ax, col in zip(a_axs, cols):
    ax.set_ylabel(col, rotation=90, fontsize=13)

# Summary figure
t_color = "b"
o_color = "r"

model_markers = ["o", "^", "s"]

result_dirs = os.listdir(config["base_dir"])

def add_data_to_figure(s_axes, a_axes, dir_path, hypoth_dict, edgecolor, alpha, tasks_list, x_locs, failed_models):
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
                #print(TASK_ID_2_STRING[int(mask_task)])

                sub_performance_target = [] 
                abl_performance_target = [] 
                sub_performance_other = [] 
                abl_performance_other = [] 

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

                    sub_performance_target.append(high_sub_acc)
                    sub_performance_other.append(low_sub_acc)
                    abl_performance_target.append(low_sub_abl_acc)
                    abl_performance_other.append(high_sub_abl_acc)
                # Add data
                # Establish one train task per col    
                if int(mask_task) == np.min(tasks_list): # first col for lower train task
                    task_col = 0
                else:
                    task_col = 1
                
                model_id = int(np.floor(idx/2)) # results are ordered according to model name, so every two results files indicate a new model

                # Subnetwork Plot, Ablate Plot
                s_ax = s_axes[task_col]
                a_ax = a_axes[task_col]

                x = x_locs + (model_id * .05)# the label locations

                if model_id == 1:
                    s_ax.set_xticks(x, ["Target", "Other"], fontsize=13)
                    a_ax.set_xticks(x, ["Target", "Other"], fontsize=13)

                if model_id in failed_models:
                    sub_target = s_ax.scatter(np.repeat([x[0]], len(sub_performance_target)), sub_performance_target, c="gray", alpha=alpha, marker=model_markers[model_id], edgecolors=edgecolor)
                    sub_other = s_ax.scatter(np.repeat([x[1]], len(sub_performance_other)), sub_performance_other, c="gray", alpha=alpha, marker=model_markers[model_id], edgecolors=edgecolor)

                    abl_target = a_ax.scatter(np.repeat([x[0]],  len(abl_performance_target)), abl_performance_target, c="gray", alpha=alpha, marker=model_markers[model_id], edgecolors=edgecolor)
                    abl_other = a_ax.scatter(np.repeat([x[1]],  len(abl_performance_other)), abl_performance_other, c="gray", alpha=alpha, marker=model_markers[model_id], edgecolors=edgecolor)

                else:
                    sub_target = s_ax.scatter(np.repeat([x[0]], len(sub_performance_target)), sub_performance_target, c=t_color, alpha=alpha, marker=model_markers[model_id], edgecolors=edgecolor)
                    sub_other = s_ax.scatter(np.repeat([x[1]], len(sub_performance_other)), sub_performance_other, c=o_color, alpha=alpha, marker=model_markers[model_id], edgecolors=edgecolor)

                    abl_target = a_ax.scatter(np.repeat([x[0]],  len(abl_performance_target)), abl_performance_target, c=t_color, alpha=alpha, marker=model_markers[model_id], edgecolors=edgecolor)
                    abl_other = a_ax.scatter(np.repeat([x[1]],  len(abl_performance_other)), abl_performance_other, c=o_color, alpha=alpha, marker=model_markers[model_id], edgecolors=edgecolor)


add_data_to_figure(s_axs, a_axs, config["base_dir"], config["hypothesized_performance"], "face", 1.0, train_tasks, np.arange(2)/4  - .05, failed_models)

# Save the figure and show
plt.tight_layout()

# Create summary legend
legend_elements = [
    s_axs[0].scatter([], [], color='black', marker=model_markers[0], label='Seed 1'),
    s_axs[0].scatter([], [], color='black', marker=model_markers[1], label='Seed 2'),
    s_axs[0].scatter([], [], color='black', marker=model_markers[2], label='Seed 3')
]

#for ax in s_axs: ax.legend(handles=legend_elements)
#for ax in a_axs: ax.legend(handles=legend_elements)

s_fig.tight_layout()
graph_path = os.path.join(config["outdir"], "sub_absolute.png")
s_fig.savefig(graph_path)

a_fig.tight_layout()
graph_path = os.path.join(config["outdir"], "abl_absolute.png")
a_fig.savefig(graph_path)