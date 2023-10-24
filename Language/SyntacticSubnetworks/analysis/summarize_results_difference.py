import yaml
import os
import copy
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

TASK_ID_2_STRING = {
    1: "Subject",
    2: "Verb",
    4: "S/P Subject",
    3: "S/P Verb",
    6: "Subject",
    7: "Verb",
    9: "S/P Subject",
    8: "S/P Verb",
    11: "Pronoun",
    12: "Antecedent",
    14: "S/P Pronoun",
    13: "S/P Antecedent",
    16: "Pronoun",
    17: "Antecedent",
    19: "S/P Pronoun",
    18: "S/P Antecedent",
}

TASK_ID_2_OVERALL_TASK = {
    1: "SV Agreement",
    2: "SV Agreement",
    6: "SV Agreement",
    7: "SV Agreement",
    11: "Anaphora",
    12: "Anaphora",
    16: "Anaphora",
    17: "Anaphora"
}

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, help='Config file containing the summarization arguments')
args = parser.parse_args()

stream = open(args.config, 'r')
config = yaml.load(stream, Loader=yaml.FullLoader)
stream.close()

# Create outdir
os.makedirs(config["outdir"], exist_ok=True)

# Create column labels, arbitrarily using singular dictionary
cols = []
sing_tasks = list(config["hypothesized_performance_sing"].keys())
sing_tasks = [int(t) for t in sing_tasks]
sing_tasks.sort()

for k in sing_tasks:
    cols.append(TASK_ID_2_STRING[int(k)]) # First col for lower train task
    fig_title = TASK_ID_2_OVERALL_TASK[int(k)]

# Establish distinguishing colors/markers
sub_color = "b"
abl_color = "r"

model_markers = ["o", "^", "s"]

# Create figure
t_fig, t_axs = plt.subplots(nrows=1, ncols=2, figsize=(6, 5))
for ax in t_axs: 
    ax.set_ylim(-1, 1)
    ax.set_xlim(-.1, .35)
    ax.axhline(y=0.0, color='gray', linestyle='--')

t_fig.suptitle(fig_title, fontsize=15)

# Set column labels
for ax, col in zip(t_axs, cols):
    ax.set_ylabel(col, rotation=90, fontsize=13)


def add_data_to_figure(axes, dir_path, hypoth_dict, edgecolor, alpha, tasks_list, x_locs, failed_models):
    
    data_for_significance = {
        "Base_Model": [],
        "Ablation": [],
        "Task": [],
        "Performance": [],
    }

    # First add singular task data to figure
    result_dirs = os.listdir(dir_path)

    # Model number determined by alph order of model name, totally arbitrary anyway
    result_dirs.sort()
    for idx, result_dir in enumerate(result_dirs):
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
                grouped_data = data.groupby("0_Model_ID")
                for model_id, grp in grouped_data:

                    low_sub_acc = grp[(grp["0_ablation"].isna()) & (grp["1_task"] == int(high_after_ablation)) & (grp["0_train"] == 0)]["2_test_acc"].item()
                    high_sub_acc = grp[(grp["0_ablation"].isna()) & (grp["1_task"] == int(low_after_ablation)) & (grp["0_train"] == 0)]["2_test_acc"].item()

                    low_sub_abl_acc = grp[(grp["0_ablation"] == "zero") & (grp["1_task"] == int(low_after_ablation)) & (grp["0_train"] == 0)]["2_test_acc"].item()
                    high_sub_abl_acc = grp[(grp["0_ablation"] == "zero") & (grp["1_task"] == int(high_after_ablation)) & (grp["0_train"] == 0)]["2_test_acc"].item()
                    
                    sub_performance.append(np.max([high_sub_acc, .25]) - np.max([low_sub_acc, .25]))
                    abl_performance.append(np.max([low_sub_abl_acc, .25]) - np.max([high_sub_abl_acc, .25]))

                # Add data
                # Establish one train task per row    
                if int(mask_task) == np.min(tasks_list): # first row for lower train task
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

# First add singular task data to figure
if "failed_models_sing" in config.keys():
    failed_models = config["failed_models_sing"]
else: 
    failed_models = []
    
sing_data_for_significance = add_data_to_figure(t_axs, config["sing_dir"], config["hypothesized_performance_sing"], "black", 1.0, sing_tasks, np.arange(2)/4  - .05, failed_models)
sing_data_for_significance["singular"] = [1] * len(sing_data_for_significance["Base_Model"])

# Then add plural task data to figure
pl_tasks = list(config["hypothesized_performance_pl"].keys())
pl_tasks = [int(t) for t in pl_tasks]
if "failed_models_pl" in config.keys():
    failed_models = config["failed_models_pl"]
else: 
    failed_models = []
pl_data_for_significance = add_data_to_figure(t_axs, config["pl_dir"], config["hypothesized_performance_pl"], "gray", 1.0, pl_tasks, np.arange(2)/4  - .025, failed_models)
pl_data_for_significance["singular"] = [0] * len(pl_data_for_significance["Base_Model"])

# Create summary legend
legend_elements = [
    t_axs[0].scatter([], [], color='black', marker=model_markers[0], label='Model 0'),
    t_axs[0].scatter([], [], color='black', marker=model_markers[1], label='Model 1'),
    t_axs[0].scatter([], [], color='black', marker=model_markers[2], label='Model 2'),
    Line2D([0], [0], color='black', lw=2, label="Sing."),
    Line2D([0], [0], color='gray', lw=2, label="Pl.")
]

#for ax in t_axs: ax.legend(handles=legend_elements)

t_fig.tight_layout()
graph_path = os.path.join(config["outdir"], "task_difference.png")
t_fig.savefig(graph_path)

sing_df = pd.DataFrame.from_dict(sing_data_for_significance)
pl_df = pd.DataFrame.from_dict(pl_data_for_significance)
pd.concat([sing_df, pl_df]).to_csv(os.path.join(config["outdir"], "analysis_data.csv"), index=False)