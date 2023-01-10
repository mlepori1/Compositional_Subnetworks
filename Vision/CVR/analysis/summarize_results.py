import yaml
import os
import copy
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

TASK_ID_2_STRING = {
    104: "Inside-Contact: Inside",
    105: "Inside-Contact: Contact",
    107: "+/- Inside",
    108: "+/- Contact",
    110: "Inside-Number: Inside",
    111: "Inside-Number: Number",
    113: "+/- Number",
    114: "+/- Inside",
    116: "Number-Contact: Contact",
    117: "Number-Contact: Number",
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

#Delete existing figures
if os.path.exists(os.path.join(config["base_dir"], "avg_mask.png")):
  os.remove(os.path.join(config["base_dir"], "avg_mask.png"))
if os.path.exists(os.path.join(config["base_dir"], "best_mask.png")):
  os.remove(os.path.join(config["base_dir"], "best_mask.png"))

rows = []
train_tasks = list(config["hypothesized_performance"].keys())
train_tasks = [int(t) for t in train_tasks]
train_tasks.sort()

for k in train_tasks:
    rows.append(TASK_ID_2_STRING[int(k)]) # First row for lower train task
    fig_title = TASK_ID_2_OVERALL_TASK[int(k)]

columns = ["Model " + str(i) for i in range(1, 4)]

# Best figure
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
for i in range(2):  
    for ax in axs[i]: 
        ax.set_ylim(0, 1)

if "Random" in config["base_dir"]:
    fig.suptitle("Random " + config["model"] + ": " + fig_title, fontsize=16)
else:
    fig.suptitle(config["model"] + ": " + fig_title, fontsize=16)

for ax, col in zip(axs[0], columns):
    ax.set_title(col)

for ax, row in zip(axs[:,0], rows):
    ax.set_ylabel(row, rotation=90, size='large')

# Avg Figure
avg_fig, avg_axs = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
for i in range(2):  
    for ax in avg_axs[i]: 
        ax.set_ylim(0, 1)
    
if "Random" in config["base_dir"]:
    avg_fig.suptitle("Random " + config["model"] + ": " + fig_title, fontsize=16)
else:
    avg_fig.suptitle(config["model"] + ": " + fig_title, fontsize=16)

for ax, col in zip(avg_axs[0], columns):
    ax.set_title(col)

for ax, row in zip(avg_axs[:,0], rows):
    ax.set_ylabel(row, rotation=90, size='large')

summary_data = []

result_dirs = os.listdir(config["base_dir"])

# Model number determined by alph order of model name, totally arbitrary anyway
result_dirs.sort()
output_dir = config['base_dir']

for idx, result_dir in enumerate(result_dirs):
    for mask_task in config["hypothesized_performance"]:
        if "T" + str(mask_task) in result_dir:

            all_low_sub_acc = []
            all_high_sub_acc = []
            all_low_abl_acc = []
            all_high_abl_acc = []

            high_after_ablation = config["hypothesized_performance"][mask_task]["high_after_ablation"]
            low_after_ablation = config["hypothesized_performance"][mask_task]["low_after_ablation"]

            data = pd.read_csv(os.path.join(output_dir, result_dir, "results.csv"))

            # Assert that all models ran 3 times
            assert len(data["0_Model_ID"].unique()) == 3

            quality_data = []
            grouped_data = data.groupby("0_Model_ID")
            for model_id, grp in grouped_data:

                # First gather results for average and std
                all_low_sub_acc.append(grp[(grp["0_ablation"].isna()) & (grp["1_task"] == int(high_after_ablation)) & (grp["0_train"] == 0)]["2_test_acc"].item())
                all_high_sub_acc.append(grp[(grp["0_ablation"].isna()) & (grp["1_task"] == int(low_after_ablation)) & (grp["0_train"] == 0)]["2_test_acc"].item())

                all_high_abl_acc.append(grp[(grp["0_ablation"] == "zero") & (grp["1_task"] == int(high_after_ablation)) & (grp["0_train"] == 0)]["2_test_acc"].item())
                all_low_abl_acc.append(grp[(grp["0_ablation"] == "zero") & (grp["1_task"] == int(low_after_ablation)) & (grp["0_train"] == 0)]["2_test_acc"].item())

                # Next calculate quality score
                # First, filter based on train task subroutine performance on validation set
                subnetwork_acc = grp[(grp["0_ablation"].isna()) & (grp["1_task"] == int(mask_task)) & (grp["0_train"] == 0)]["2_test_acc"].item()
                if subnetwork_acc < .9: 
                    quality_score = 0.0
                else:
                    # Next, compute ablation quality using zero-ablation
                    low_sub_acc = grp[(grp["0_ablation"].isna()) & (grp["1_task"] == int(high_after_ablation)) & (grp["0_train"] == 0)]["2_test_acc"].item()
                    high_sub_acc = grp[(grp["0_ablation"].isna()) & (grp["1_task"] == int(low_after_ablation)) & (grp["0_train"] == 0)]["2_test_acc"].item()

                    # Next, compute ablation quality using zero-ablation
                    high_abl_acc = grp[(grp["0_ablation"] == "zero") & (grp["1_task"] == int(high_after_ablation)) & (grp["0_train"] == 0)]["2_test_acc"].item()
                    low_abl_acc = grp[(grp["0_ablation"] == "zero") & (grp["1_task"] == int(low_after_ablation)) & (grp["0_train"] == 0)]["2_test_acc"].item()
                    quality_score = np.max([high_abl_acc, .25]) - np.max([low_abl_acc, .25]) # Max is 0.75, Min is 0.0

                grp["Quality Score"] = len(grp) * [quality_score]
                grp["0_Model_ID"] = len(grp) * [model_id]

                quality_data.append(grp)

            quality_data = pd.concat(quality_data)


            max_quality = np.max(quality_data["Quality Score"])
            best_masked_model = quality_data[quality_data["Quality Score"] == max_quality]["0_Model_ID"].iloc[0]
            best_result = quality_data[quality_data["0_Model_ID"] == best_masked_model]

            # Next, compute ablation quality using zero-ablation
            low_sub_acc = best_result[(best_result["0_ablation"].isna()) & (best_result["1_task"] == int(high_after_ablation)) & (best_result["0_train"] == 0)]["2_test_acc"].item()
            high_sub_acc = best_result[(best_result["0_ablation"].isna()) & (best_result["1_task"] == int(low_after_ablation)) & (best_result["0_train"] == 0)]["2_test_acc"].item()

            # Next, compute ablation quality using zero-ablation
            high_abl_acc = best_result[(best_result["0_ablation"] == "zero") & (best_result["1_task"] == int(high_after_ablation)) & (best_result["0_train"] == 0)]["2_test_acc"].item()
            low_abl_acc = best_result[(best_result["0_ablation"] == "zero") & (best_result["1_task"] == int(low_after_ablation)) & (best_result["0_train"] == 0)]["2_test_acc"].item()

            # Build the plots
            # Establish one train task per row    
            if int(mask_task) == np.min(train_tasks): # first row for lower train task
                task_row = 0
                task_column = int(np.floor(idx/2)) # results are ordered according to model name, so every two results files indicate a new model

                # Best plot first
                ax = axs[task_row, task_column]
                labels = ["Subnetwork", "Ablation"]
                task_1_performance = [high_sub_acc, low_abl_acc]
                task_2_performance = [low_sub_acc, high_abl_acc]
                x = np.arange(len(labels))  # the label locations
                width = 0.3  # the width of the bars

                rects1 = ax.bar(x - width/2, task_1_performance, width, label=TASK_ID_2_STRING[int(low_after_ablation)])
                rects2 = ax.bar(x + width/2, task_2_performance, width, label=TASK_ID_2_STRING[int(high_after_ablation)])

                ax.set_xticks(x, labels)
                ax.legend()

                # Average plot
                ax = avg_axs[task_row, task_column]
                labels = ["Subnetwork", "Ablation"]
                task_1_performance = [np.mean(all_high_sub_acc), np.mean(all_low_abl_acc)]
                task_1_yerr = [np.std(all_high_sub_acc), np.std(all_low_abl_acc)]
                task_2_performance = [np.mean(all_low_sub_acc), np.mean(all_high_abl_acc)]
                task_2_yerr = [np.std(all_low_sub_acc), np.std(all_high_abl_acc)]

                x = np.arange(len(labels))  # the label locations
                width = 0.3  # the width of the bars

                rects1 = ax.bar(x - width/2, task_1_performance, width, yerr=task_1_yerr, label=TASK_ID_2_STRING[int(low_after_ablation)])
                rects2 = ax.bar(x + width/2, task_2_performance, width, yerr=task_2_yerr, label=TASK_ID_2_STRING[int(high_after_ablation)])

                ax.set_xticks(x, labels)
                ax.legend()
            else:
                task_row = 1
                task_column = int(np.floor(idx/2)) # results are ordered according to model name, so every two results files indicate a new model

                # Best plot first
                ax = axs[task_row, task_column]

                labels = ["Subnetwork", "Ablation"]
                task_2_performance = [high_sub_acc, low_abl_acc]
                task_1_performance = [low_sub_acc, high_abl_acc]
                x = np.arange(len(labels))  # the label locations
                width = 0.3  # the width of the bars

                rects1 = ax.bar(x - width/2, task_1_performance, width, label=TASK_ID_2_STRING[int(high_after_ablation)])
                rects2 = ax.bar(x + width/2, task_2_performance, width, label=TASK_ID_2_STRING[int(low_after_ablation)])

                ax.set_xticks(x, labels)
                ax.legend()

                # Avg plot
                ax = avg_axs[task_row, task_column]
                labels = ["Subnetwork", "Ablation"]
                task_2_performance = [np.mean(all_high_sub_acc), np.mean(all_low_abl_acc)]
                task_2_yerr = [np.std(all_high_sub_acc), np.std(all_low_abl_acc)]
                task_1_performance = [np.mean(all_low_sub_acc), np.mean(all_high_abl_acc)]
                task_1_yerr = [np.std(all_low_sub_acc), np.std(all_high_abl_acc)]

                x = np.arange(len(labels))  # the label locations
                width = 0.3  # the width of the bars

                rects1 = ax.bar(x - width/2, task_1_performance, width, yerr=task_1_yerr, label=TASK_ID_2_STRING[int(high_after_ablation)])
                rects2 = ax.bar(x + width/2, task_2_performance, width, yerr=task_2_yerr, label=TASK_ID_2_STRING[int(low_after_ablation)])
                ax.set_xticks(x, labels)
                ax.legend()

# Save the figure and show
plt.tight_layout()

graph_path = os.path.join(output_dir, "best_mask.png")
fig.savefig(graph_path)

graph_path = os.path.join(output_dir, "avg_mask.png")
avg_fig.savefig(graph_path)