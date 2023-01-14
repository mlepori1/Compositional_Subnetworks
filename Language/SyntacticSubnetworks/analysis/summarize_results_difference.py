import yaml
import os
import copy
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

TASK_ID_2_STRING = {
    1: "SV Agreement Singular: Subject",
    2: "SV Agreement Singular: Verb",
    4: "S/P Subject",
    3: "S/P Verb",
    6: "SV Agreement Plural: Subject",
    7: "SV Agreement Plural: Verb",
    9: "S/P Subject",
    8: "S/P Verb",
    11: "Anaphora Singular: Pronoun",
    12: "Anaphora Singular: Antecedent",
    14: "S/P Pronoun",
    13: "S/P Antecedent",
    16: "Anaphora Plural: Pronoun",
    17: "Anaphora Plural: Antecedent",
    19: "S/P Pronoun",
    18: "S/P Antecedent",
}

TASK_ID_2_OVERALL_TASK = {
    1: "SV Agreement Singular",
    2: "SV Agreement Singular",
    6: "SV Agreement Plural",
    7: "SV Agreement Plural",
    11: "Anaphora Singular",
    12: "Anaphora Singular",
    16: "Anaphora Plural",
    17: "Anaphora Plural"
}

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, help='Config file containing the summarization arguments')
args = parser.parse_args()

stream = open(args.config, 'r')
config = yaml.load(stream, Loader=yaml.FullLoader)
stream.close()

#Delete existing figures
if os.path.exists(os.path.join(config["base_dir"], "disag_difference.png")):
  os.remove(os.path.join(config["base_dir"], "disag_difference.png"))
if os.path.exists(os.path.join(config["base_dir"], "summary_difference.png")):
  os.remove(os.path.join(config["base_dir"], "summary_difference.png"))
if os.path.exists(os.path.join(config["base_dir"], "task_difference.png")):
  os.remove(os.path.join(config["base_dir"], "task_difference.png"))
if os.path.exists(os.path.join(config["base_dir"], "avg_mask.png")):
  os.remove(os.path.join(config["base_dir"], "avg_mask.png"))
if os.path.exists(os.path.join(config["base_dir"], "best_mask.png")):
  os.remove(os.path.join(config["base_dir"], "best_mask.png"))

if "failed_models" in config.keys():
    failed_models = config["failed_models"]
else: 
    failed_models = []

rows = []
train_tasks = list(config["hypothesized_performance"].keys())
train_tasks = [int(t) for t in train_tasks]
train_tasks.sort()

for k in train_tasks:
    rows.append(TASK_ID_2_STRING[int(k)]) # First row for lower train task
    fig_title = TASK_ID_2_OVERALL_TASK[int(k)]

columns = ["Model " + str(i) for i in range(1, 4)]

# Disagregated figure
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
for i in range(2):  
    for ax in axs[i]: 
        ax.set_ylim(-.75, .75)
        ax.set_xlim(-.5, 1)
        ax.axhline(y=0.0, color='gray', linestyle='--')

if "Random" in config["base_dir"]:
    fig_title = "Random " + config["model"] + ": " + fig_title + " (Target - Other)"
    fig.suptitle(fig_title, fontsize=16)
else:
    fig_title = config["model"] + ": " + fig_title + " (Target - Other)"
    fig.suptitle(fig_title, fontsize=16)

for ax, col in zip(axs[0], columns):
    ax.set_title(col)

for ax, row in zip(axs[:,0], rows):
    ax.set_ylabel(row, rotation=90, size='large')

# Task-Agg'd figure
t_fig, t_axs = plt.subplots(nrows=2, ncols=1, figsize=(8, 10))
for ax in t_axs: 
    ax.set_ylim(-1, 1)
    ax.set_xlim(-.5, 1)
    ax.axhline(y=0.0, color='gray', linestyle='--')

t_fig.suptitle(fig_title, fontsize=12)

for ax, row in zip(t_axs, rows):
    ax.set_ylabel(row, rotation=90, size='large')

# Summary figure
sub_color = "b"
abl_color = "r"

model_markers = ["o", "^", "s"]

sum_fig, sum_ax = plt.subplots(figsize=(5, 5))
sum_ax.set_ylim(-1, 1)
sum_ax.set_xlim(-.5, 1)
sum_ax.axhline(y=0.0, color='gray', linestyle='--')

sum_ax.set_title(fig_title, fontsize=12)

result_dirs = os.listdir(config["base_dir"])

# Model number determined by alph order of model name, totally arbitrary anyway
result_dirs.sort()
output_dir = config['base_dir']

for idx, result_dir in enumerate(result_dirs):
    for mask_task in config["hypothesized_performance"]:
        if "T" + str(mask_task) in result_dir:

            all_sub_diffs = [] # Difference between subnet performance on the two tasks 
            all_abl_diffs = [] # Difference between the ablated network on the two tasks

            high_after_ablation = config["hypothesized_performance"][mask_task]["high_after_ablation"]
            low_after_ablation = config["hypothesized_performance"][mask_task]["low_after_ablation"]

            data = pd.read_csv(os.path.join(output_dir, result_dir, "results.csv"))

            # Assert that all models ran 3 times
            assert len(data["0_Model_ID"].unique()) == 3

            # Model ID here corresponds to each run of the mask search algorithm
            grouped_data = data.groupby("0_Model_ID")
            for model_id, grp in grouped_data:

                # First gather results for average and std
                low_sub_acc = grp[(grp["0_ablation"].isna()) & (grp["1_task"] == int(high_after_ablation)) & (grp["0_train"] == 0)]["2_test_acc"].item()
                high_sub_acc = grp[(grp["0_ablation"].isna()) & (grp["1_task"] == int(low_after_ablation)) & (grp["0_train"] == 0)]["2_test_acc"].item()

                low_sub_abl_acc = grp[(grp["0_ablation"] == "zero") & (grp["1_task"] == int(low_after_ablation)) & (grp["0_train"] == 0)]["2_test_acc"].item()
                high_sub_abl_acc = grp[(grp["0_ablation"] == "zero") & (grp["1_task"] == int(high_after_ablation)) & (grp["0_train"] == 0)]["2_test_acc"].item()
                
                all_sub_diffs.append(np.max([high_sub_acc, .25]) - np.max([low_sub_acc, .25]))
                all_abl_diffs.append(np.max([low_sub_abl_acc, .25]) - np.max([high_sub_abl_acc, .25]))


            # Build the plots
            # Establish one train task per row    
            if int(mask_task) == np.min(train_tasks): # first row for lower train task
                task_row = 0
                task_column = int(np.floor(idx/2)) # results are ordered according to model name, so every two results files indicate a new model

                # disagregated plot first
                ax = axs[task_row, task_column]
                sub_performance = all_sub_diffs
                abl_performance = all_abl_diffs
                x = np.arange(2)/2  # the label locations

                ax.set_xticks(x, ["Subnetwork", "Ablation"])

                sub = ax.scatter(np.repeat([x[0]], len(sub_performance)), sub_performance)
                abl = ax.scatter(np.repeat([x[1]],  len(abl_performance)), abl_performance)

                # Task aggregated Plot
                ax = t_axs[task_row]

                x = np.arange(2)/2  - .05 + (task_column * .05)# the label locations

                if task_column == 1:
                    ax.set_xticks(x, ["Subnetwork", "Ablation"])

                if task_column in failed_models:
                    sub = ax.scatter(np.repeat([x[0]], len(sub_performance)), sub_performance, c="gray", alpha=0.5, marker=model_markers[task_column], label="Model " + str(task_column))
                    abl = ax.scatter(np.repeat([x[1]],  len(abl_performance)), abl_performance, c="gray", alpha=0.5, marker=model_markers[task_column], label="Model " + str(task_column))
                else:
                    sub = ax.scatter(np.repeat([x[0]], len(sub_performance)), sub_performance, c=sub_color, marker=model_markers[task_column], label="Model " + str(task_column))
                    abl = ax.scatter(np.repeat([x[1]],  len(abl_performance)), abl_performance, c=abl_color, marker=model_markers[task_column], label="Model " + str(task_column))

                # Summary Plot
                x = np.arange(2)/2  - .05 + (task_column * .05)# the label locations

                if task_column == 1:
                    sum_ax.set_xticks(x, ["Subnetwork", "Ablation"])

                if task_column in failed_models:
                    sub = sum_ax.scatter(np.repeat([x[0]], len(sub_performance)), sub_performance, c="gray", alpha=0.5, marker=model_markers[task_column], label="Model " + str(task_column))
                    abl = sum_ax.scatter(np.repeat([x[1]],  len(abl_performance)), abl_performance, c="gray", alpha=0.5, marker=model_markers[task_column], label="Model " + str(task_column))
                else:
                    sub = sum_ax.scatter(np.repeat([x[0]], len(sub_performance)), sub_performance, c=sub_color, marker=model_markers[task_column], label="Model " + str(task_column))
                    abl = sum_ax.scatter(np.repeat([x[1]],  len(abl_performance)), abl_performance, c=abl_color, marker=model_markers[task_column], label="Model " + str(task_column))

            else:
                task_row = 1
                task_column = int(np.floor(idx/2)) # results are ordered according to model name, so every two results files indicate a new model

                # disaggregated plot first
                ax = axs[task_row, task_column]
                sub_performance = all_sub_diffs
                abl_performance = all_abl_diffs
                x = np.arange(2)/2  # the label locations

                ax.set_xticks(x, ["Subnetwork", "Ablation"])

                sub = ax.scatter(np.repeat([x[0]], len(sub_performance)), sub_performance)
                abl = ax.scatter(np.repeat([x[1]],  len(abl_performance)), abl_performance)

                # Task aggregated Plot
                ax = t_axs[task_row]

                x = np.arange(2)/2  - .05 + (task_column * .05)# the label locations

                if task_column == 1:
                    ax.set_xticks(x, ["Subnetwork", "Ablation"])

                if task_column in failed_models:
                    sub = ax.scatter(np.repeat([x[0]], len(sub_performance)), sub_performance, c="gray", alpha=0.5, marker=model_markers[task_column], label="Model " + str(task_column))
                    abl = ax.scatter(np.repeat([x[1]],  len(abl_performance)), abl_performance, c="gray", alpha=0.5, marker=model_markers[task_column], label="Model " + str(task_column))
                else:
                    sub = ax.scatter(np.repeat([x[0]], len(sub_performance)), sub_performance, c=sub_color, marker=model_markers[task_column], label="Model " + str(task_column))
                    abl = ax.scatter(np.repeat([x[1]],  len(abl_performance)), abl_performance, c=abl_color, marker=model_markers[task_column], label="Model " + str(task_column))

                # Summary Plot
                x = np.arange(2)/2  - .05 + (task_column * .05)# the label locations

                if task_column == 1:
                    sum_ax.set_xticks(x, ["Subnetwork", "Ablation"])

                if task_column in failed_models:
                    sub = sum_ax.scatter(np.repeat([x[0]], len(sub_performance)), sub_performance, c="gray", alpha=0.5, marker=model_markers[task_column], label="Model " + str(task_column))
                    abl = sum_ax.scatter(np.repeat([x[1]],  len(abl_performance)), abl_performance, c="gray", alpha=0.5, marker=model_markers[task_column], label="Model " + str(task_column))
                else:
                    sub = sum_ax.scatter(np.repeat([x[0]], len(sub_performance)), sub_performance, c=sub_color, marker=model_markers[task_column], label="Model " + str(task_column))
                    abl = sum_ax.scatter(np.repeat([x[1]],  len(abl_performance)), abl_performance, c=abl_color, marker=model_markers[task_column], label="Model " + str(task_column))

# Save the figure and show
plt.tight_layout()

graph_path = os.path.join(output_dir, "disag_difference.png")
fig.savefig(graph_path)

# Create summary legend
legend_elements = [
    sum_ax.scatter([], [], color='black', marker=model_markers[0], label='Model 0'),
    sum_ax.scatter([], [], color='black', marker=model_markers[1], label='Model 1'),
    sum_ax.scatter([], [], color='black', marker=model_markers[2], label='Model 2')
]

for ax in t_axs: ax.legend(handles=legend_elements)

graph_path = os.path.join(output_dir, "task_difference.png")
t_fig.savefig(graph_path)

sum_ax.legend(handles=legend_elements)

graph_path = os.path.join(output_dir, "summary_difference.png")
sum_fig.savefig(graph_path)