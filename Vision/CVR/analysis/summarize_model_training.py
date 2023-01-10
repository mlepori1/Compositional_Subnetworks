import yaml
import os
import copy
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

TASK_ID_2_OVERALL_TASK = {
    103: "Inside-Contact",
    109: "Inside-Number",
    115: "Number-Contact",
}

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, help='Config file containing the summarization arguments')
args = parser.parse_args()

stream = open(args.config, 'r')
config = yaml.load(stream, Loader=yaml.FullLoader)
stream.close()

columns = ["Model " + str(i) for i in range(1, 4)]

# Best figure
fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 10))

fig.suptitle(config["model"] + ": " + TASK_ID_2_OVERALL_TASK[int(config["task"])], fontsize=16)

for ax in axs: 
    ax.set_ylim(0, 1)

for ax, col in zip(axs, columns):
    ax.set_title(col)

result_dirs = os.listdir(config["base_dir"])

# Model number determined by alph order of model name, totally arbitrary anyway
output_dir = config['base_dir']

data = pd.read_csv(os.path.join(output_dir, "results.csv"))
ids = list(data["0_Model_ID"].unique())
ids.sort()

grouped_data = data.groupby("0_Model_ID")
for model_id, grp in grouped_data:

    # First gather results for average and std
    acc = grp[(grp["0_ablation"].isna()) & (grp["1_task"] == int(config["task"])) & (grp["0_train"] == 0)]["2_test_acc"].item()
    print(acc)
    axs[ids.index(model_id)].bar(0, acc)

# Save the figure and show
plt.tight_layout()

graph_path = os.path.join(output_dir, "Model_Accuracies.png")
fig.savefig(graph_path)