import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

dirs = os.listdir(".")

models = ["model1", "model2", "model3"]
tasks = ["T1", "T2"]

for model in models:
    for task in tasks:
        results = None
        for d in dirs:
            if model in d and task in d:
                results = pd.read_csv(d + "/results.csv")
    

        t3_sub = []
        t3_ablate = []
        t4_sub = []
        t4_ablate = []

        t3_sub = results[(pd.isna(results["0_ablation"])) & (results["1_task"] == 3)]["2_test_acc"].tolist()
        t3_ablate = results[(results["0_ablation"] == "zero") & (results["1_task"] == 3)]["2_test_acc"].tolist()
 
        t4_sub = results[pd.isna(results["0_ablation"]) & (results["1_task"] == 4)]["2_test_acc"].tolist()
        t4_ablate = results[(results["0_ablation"] == "zero") & (results["1_task"] == 4)]["2_test_acc"].tolist()
    
        t3_sub_std = np.std(t3_sub)
        t4_sub_std = np.std(t4_sub)
        t3_ablate_std = np.std(t3_ablate)
        t4_ablate_std = np.std(t4_ablate)

        t3_sub_mean = np.mean(t3_sub)
        t4_sub_mean = np.mean(t4_sub)
        t3_ablate_mean = np.mean(t3_ablate)
        t4_ablate_mean = np.mean(t4_ablate)


        t3_name ="OOO_VERB"
        t4_name = "OOO_Subj"
        if task == "T1":
            taskname = "Subject_Singular"
        if task == "T2":
            taskname = "Verb_Singular"

        # Build the plot
        fig, ax = plt.subplots()
        ax.bar(range(2), [t3_sub_mean, t4_sub_mean], yerr=[t3_sub_std, t4_sub_std], align='center', alpha=0.5, ecolor='black', capsize=10)
        ax.set_ylabel('Accuracy')
        ax.set_xticks(range(2))
        ax.set_xticklabels([t3_name, t4_name])
        ax.set_title(taskname + ' Subnet Performance ' + model)
        ax.yaxis.grid(True)
        # Save the figure and show
        plt.tight_layout()
        plt.savefig(model + taskname + "_sub.png")
        plt.show()

        # Build the plot
        fig, ax = plt.subplots()
        ax.bar(range(2), [t3_ablate_mean, t4_ablate_mean], yerr=[t3_ablate_std, t4_ablate_std], align='center', alpha=0.5, ecolor='black', capsize=10)
        ax.set_ylabel('Accuracy')
        ax.set_xticks(range(2))
        ax.set_xticklabels([t3_name, t4_name])
        ax.set_title(taskname + ' Ablate Performance ' + model)
        ax.yaxis.grid(True)
        # Save the figure and show
        plt.tight_layout()
        plt.savefig(model + taskname + "_ablate.png")
        plt.show()

        # Build the plot
        fig, ax = plt.subplots()
        ax.bar(range(2), [t3_sub[0], t4_sub[0]], align='center', alpha=0.5, ecolor='black', capsize=10)
        ax.set_ylabel('Accuracy')
        ax.set_xticks(range(2))
        ax.set_xticklabels([t3_name, t4_name])
        ax.set_title(taskname + ' Subnet Performance ' + model)
        ax.yaxis.grid(True)
        # Save the figure and show
        plt.tight_layout()
        plt.savefig(model + taskname + "_sub_first.png")
        plt.show()

        # Build the plot
        fig, ax = plt.subplots()
        ax.bar(range(2), [t3_ablate[0], t4_ablate[0]], align='center', alpha=0.5, ecolor='black', capsize=10)
        ax.set_ylabel('Accuracy')
        ax.set_xticks(range(2))
        ax.set_xticklabels([t3_name, t4_name])
        ax.set_title(taskname + ' Ablate Performance ' + model)
        ax.yaxis.grid(True)
        # Save the figure and show
        plt.tight_layout()
        plt.savefig(model + taskname + "_ablate_first.png")
        plt.show()
