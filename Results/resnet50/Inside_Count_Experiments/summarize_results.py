import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

dirs = os.listdir(".")

models = ["model1", "model2", "model3"]
tasks = ["110", "111"]
stages = ["1stage", "2stage", "3stage", "5stage"]

for model in models:
    for task in tasks:
        # Get all results dfs

        dfs = []
        for d in dirs:
            if model in d and task in d:
                dfs.append(pd.read_csv(d + "/results.csv"))
        
        results = pd.concat(dfs)

        # Create a unique identifier for every subnet
        results["unique_subnet"] = results["0_exp_dir"] + results["0_Model_ID"].astype(str)

        inside_sub = []
        inside_ablate = []
        count_sub = []
        count_ablate = []
        
        inside_pairwise = []
        count_pairwise = []

        best_inside_sub = 0
        best_inside_ablate = 0

        best_count_sub = 0
        best_count_ablate = 0

        # INSIDE HIGH PERFORMANCE PARTITION = 114
        # COUNT HIGH PERFORMANCE PARTITION = 113
        if task == "110": # Inside subnet
            unique_ids = results[results["2_val_acc"] > .9]["unique_subnet"]
            
            results = results.loc[results['unique_subnet'].isin(unique_ids)]
            best_run = results[results["0_ablation"] == "zero"]
            best_run = best_run[best_run["1_task"] >= 113]
            best_run["test_diffs"] = best_run["2_test_acc"].diff()

            best_run = best_run[best_run["1_task"] == 114]
            best_result = best_run[best_run["test_diffs"] == np.min(best_run["test_diffs"])] # 114 should be high when you ablate contact, 113 when you ablate inside, so 114-113 should be -
            best_id = best_result["0_Model_ID"].item()
            best_lr = best_result["3_lr"].item()
            best_l0_init = best_result["3_l0_init"].item()
            best_l0_stages = best_result["3_l0_stages"].item()
            best_test_l0 = best_result["2_test_L0"].item()
            best_subnet = best_result["unique_subnet"].item()

            res_set = results[(results["3_lr"] == best_lr) & (results["3_l0_init"] == best_l0_init) & (results["3_l0_stages"] == best_l0_stages)]
            for model_id in res_set["0_Model_ID"].unique():
                id_set = res_set[res_set["0_Model_ID"] == model_id]
                zero_set = id_set[id_set["0_ablation"] == "zero"]
                sub_set = id_set[(id_set["0_ablation"] != "zero") & (id_set["0_ablation"] != "random")]
                inside_sub.append(sub_set[sub_set["1_task"] == 114]["2_test_acc"].item()) 
                count_sub.append(sub_set[sub_set["1_task"] == 113]["2_test_acc"].item()) 

                inside_ablate.append(zero_set[zero_set["1_task"] == 114]["2_test_acc"].item()) 
                count_ablate.append(zero_set[zero_set["1_task"] == 113]["2_test_acc"].item()) 

                inside_pairwise.append(sub_set[sub_set["1_task"] == 114]["2_test_acc"].item() - zero_set[zero_set["1_task"] == 114]["2_test_acc"].item())
                count_pairwise.append(sub_set[sub_set["1_task"] == 113]["2_test_acc"].item() - zero_set[zero_set["1_task"] == 113]["2_test_acc"].item())

                if id_set["unique_subnet"].unique().item() == best_subnet:
                    best_inside_sub  = sub_set[sub_set["1_task"] == 114]["2_test_acc"].item()
                    best_count_sub = sub_set[sub_set["1_task"] == 113]["2_test_acc"].item()
                    best_inside_ablate = zero_set[zero_set["1_task"] == 114]["2_test_acc"].item()
                    best_count_ablate = zero_set[zero_set["1_task"] == 113]["2_test_acc"].item()



            inside_sub_std = np.std(inside_sub)
            count_sub_std = np.std(count_sub)
            inside_ablate_std = np.std(inside_ablate)
            count_ablate_std = np.std(count_ablate)
            inside_pair_std = np.std(inside_pairwise)
            count_pair_std = np.std(count_pairwise)

            inside_sub_mean = np.mean(inside_sub)
            count_sub_mean = np.mean(count_sub)
            inside_ablate_mean = np.mean(inside_ablate)
            count_ablate_mean = np.mean(count_ablate)
            inside_pair_mean = np.mean(inside_pairwise)
            count_pair_mean = np.mean(count_pairwise)

            # Build the plot
            fig, ax = plt.subplots()
            ax.bar(range(2), [inside_sub_mean, count_sub_mean], yerr=[inside_sub_std, count_sub_std], align='center', alpha=0.5, ecolor='black', capsize=10)
            ax.set_ylabel('Accuracy')
            ax.set_xticks(range(2))
            ax.set_xticklabels(["Inside", "Count"])
            ax.set_title('Inside Subnetwork Performance ' + model)
            ax.yaxis.grid(True)
            ax.annotate(" ".join(best_l0_stages), xy=(-0.2,.1))
            ax.annotate("L0 Init: " + str(best_l0_init), xy=(.5, .2))
            # Save the figure and show
            plt.tight_layout()
            plt.savefig(model + ' Inside_Subnetwork_Sub_Performance.png')
            plt.show()

            # Build the plot
            fig, ax = plt.subplots()
            ax.bar(range(2), [inside_ablate_mean, count_ablate_mean], yerr=[inside_ablate_std, count_ablate_std], align='center', alpha=0.5, ecolor='black', capsize=10)
            ax.set_ylabel('Accuracy')
            ax.set_xticks(range(2))
            ax.set_xticklabels(["Inside", "Count"])
            ax.set_title('Ablated Inside Performance ' + model)
            ax.yaxis.grid(True)
            ax.annotate(" ".join(best_l0_stages), xy=(-0.2,.1))
            ax.annotate("L0 Init: " + str(best_l0_init), xy=(.5, .2))

            # Save the figure and show
            plt.tight_layout()
            plt.savefig(model + ' Inside_Subnetwork_Ablate_Performance.png')
            plt.show()

            # Build the plot
            fig, ax = plt.subplots()
            ax.bar(range(2), [inside_pair_mean, count_pair_mean], yerr=[inside_pair_std, count_pair_std], align='center', alpha=0.5, ecolor='black', capsize=10)
            ax.set_ylabel('Accuracy')
            ax.set_xticks(range(2))
            ax.set_xticklabels(["Inside", "Count"])
            ax.set_title('Pairwise (Sub - Ablate) Inside Performance ' + model)
            ax.yaxis.grid(True)
            ax.annotate(" ".join(best_l0_stages), xy=(-0.2,.1))
            ax.annotate("L0 Init: " + str(best_l0_init), xy=(.5, .2))
            # Save the figure and show
            plt.tight_layout()
            plt.savefig(model + ' Inside_Pairwise_Performance.png')
            plt.show()

            # Build the plot
            fig, ax = plt.subplots()
            ax.bar(range(2), [best_inside_sub, best_count_sub], align='center', alpha=0.5, ecolor='black', capsize=10)
            ax.set_ylabel('Accuracy')
            ax.set_xticks(range(2))
            ax.set_xticklabels(["Inside", "Count"])
            ax.set_title('Best Inside Subnetwork Performance ' + model)
            ax.yaxis.grid(True)
            ax.annotate(" ".join(best_l0_stages), xy=(-0.2,.1))
            ax.annotate("L0 Init: " + str(best_l0_init), xy=(.5, .2))

            # Save the figure and show
            plt.tight_layout()
            plt.savefig(model + ' Best_Inside_Subnetwork_Performance.png')
            plt.show()

            # Build the plot
            fig, ax = plt.subplots()
            ax.bar(range(2), [best_inside_ablate, best_count_ablate], align='center', alpha=0.5, ecolor='black', capsize=10)
            ax.set_ylabel('Accuracy')
            ax.set_xticks(range(2))
            ax.set_xticklabels(["Inside", "Count"])
            ax.set_title('Best Inside Subnetwork Ablated Performance ' + model)
            ax.yaxis.grid(True)
            ax.annotate(" ".join(best_l0_stages), xy=(-0.2,.1))
            ax.annotate("L0 Init: " + str(best_l0_init), xy=(.5, .2))

            # Save the figure and show
            plt.tight_layout()
            plt.savefig(model + ' Best_Inside_Subnetwork_Ablated_Performance.png')
            plt.show()

            dfs = []

        elif task == "111": # Count subnet
            unique_ids = results[results["2_val_acc"] > .9]["unique_subnet"]
            results = results.loc[results['unique_subnet'].isin(unique_ids)]
            best_run = results[results["0_ablation"] == "zero"]
            best_run = best_run[best_run["1_task"] >= 113]
            best_run["test_diffs"] = best_run["2_test_acc"].diff()
            best_run = best_run[best_run["1_task"] == 114]
            best_result = best_run[best_run["test_diffs"] == np.max(best_run["test_diffs"])] # 114 should yeild high performance when ablating count, 113 when ablating inside, difference should be positive then
            best_id = best_result["0_Model_ID"].item()
            best_lr = best_result["3_lr"].item()
            best_l0_init = best_result["3_l0_init"].item()
            best_l0_stages = best_result["3_l0_stages"].item()
            best_test_l0 = best_result["2_test_L0"].item()
            best_subnet = best_result["unique_subnet"].item()

            res_set = results[(results["3_lr"] == best_lr) & (results["3_l0_init"] == best_l0_init) & (results["3_l0_stages"] == best_l0_stages)]
            for model_id in res_set["0_Model_ID"].unique():
                id_set = res_set[res_set["0_Model_ID"] == model_id]

                zero_set = id_set[id_set["0_ablation"] == "zero"]
                sub_set = id_set[(id_set["0_ablation"] != "zero") & (id_set["0_ablation"] != "random")]
                inside_sub.append(sub_set[sub_set["1_task"] == 114]["2_test_acc"].item()) 
                count_sub.append(sub_set[sub_set["1_task"] == 113]["2_test_acc"].item()) 

                inside_ablate.append(zero_set[zero_set["1_task"] == 114]["2_test_acc"].item()) 
                count_ablate.append(zero_set[zero_set["1_task"] == 113]["2_test_acc"].item()) 

                inside_pairwise.append(sub_set[sub_set["1_task"] == 114]["2_test_acc"].item() - zero_set[zero_set["1_task"] == 114]["2_test_acc"].item())
                count_pairwise.append(sub_set[sub_set["1_task"] == 113]["2_test_acc"].item() - zero_set[zero_set["1_task"] == 113]["2_test_acc"].item())

                if id_set["unique_subnet"].unique().item() == best_subnet:
                    best_inside_sub  = sub_set[sub_set["1_task"] == 114]["2_test_acc"].item()
                    best_count_sub = sub_set[sub_set["1_task"] == 113]["2_test_acc"].item()
                    best_inside_ablate = zero_set[zero_set["1_task"] == 114]["2_test_acc"].item()
                    best_count_ablate = zero_set[zero_set["1_task"] == 113]["2_test_acc"].item()

            inside_sub_std = np.std(inside_sub)
            count_sub_std = np.std(count_sub)
            inside_ablate_std = np.std(inside_ablate)
            count_ablate_std = np.std(count_ablate)
            inside_pair_std = np.std(inside_pairwise)
            count_pair_std = np.std(count_pairwise)

            inside_sub_mean = np.mean(inside_sub)
            count_sub_mean = np.mean(count_sub)
            inside_ablate_mean = np.mean(inside_ablate)
            count_ablate_mean = np.mean(count_ablate)
            inside_pair_mean = np.mean(inside_pairwise)
            count_pair_mean = np.mean(count_pairwise)

            
            # Build the plot
            fig, ax = plt.subplots()
            ax.bar(range(2), [inside_sub_mean, count_sub_mean], yerr=[inside_sub_std, count_sub_std], align='center', alpha=0.5, ecolor='black', capsize=10)
            ax.set_ylabel('Accuracy')
            ax.set_xticks(range(2))
            ax.set_xticklabels(["Inside", "Count"])
            ax.set_title('Count Subnetwork Performance ' + model)
            ax.yaxis.grid(True)
            ax.annotate(" ".join(best_l0_stages), xy=(-0.2,.1))
            ax.annotate("L0 Init: " + str(best_l0_init), xy=(.5, .2))
            # Save the figure and show
            plt.tight_layout()
            plt.savefig(model + ' Count_Subnetwork_Sub_Performance.png')
            plt.show()

            # Build the plot
            fig, ax = plt.subplots()
            ax.bar(range(2), [inside_ablate_mean, count_ablate_mean], yerr=[inside_ablate_std, count_ablate_std], align='center', alpha=0.5, ecolor='black', capsize=10)
            ax.set_ylabel('Accuracy')
            ax.set_xticks(range(2))
            ax.set_xticklabels(["Inside", "Count"])
            ax.set_title('Ablated Count Performance ' + model)
            ax.yaxis.grid(True)
            ax.annotate(" ".join(best_l0_stages), xy=(-0.2,.1))
            ax.annotate("L0 Init: " + str(best_l0_init), xy=(.5, .2))
            # Save the figure and show
            plt.tight_layout()
            plt.savefig(model + ' Count_Subnetwork_Ablate_Performance.png')
            plt.show()

            # Build the plot
            fig, ax = plt.subplots()
            ax.bar(range(2), [inside_pair_mean, count_pair_mean], yerr=[inside_pair_std, count_pair_std], align='center', alpha=0.5, ecolor='black', capsize=10)
            ax.set_ylabel('Accuracy')
            ax.set_xticks(range(2))
            ax.set_xticklabels(["Inside", "Count"])
            ax.set_title('Pairwise (Sub - Ablate) Count Performance ' + model)
            ax.yaxis.grid(True)
            ax.annotate(" ".join(best_l0_stages), xy=(-0.2,.1))
            ax.annotate("L0 Init: " + str(best_l0_init), xy=(.5, .2))
            # Save the figure and show
            plt.tight_layout()
            plt.savefig(model + ' Count_Pairwise_Performance.png')
            plt.show()

            # Build the plot
            fig, ax = plt.subplots()
            ax.bar(range(2), [best_inside_sub, best_count_sub], align='center', alpha=0.5, ecolor='black', capsize=10)
            ax.set_ylabel('Accuracy')
            ax.set_xticks(range(2))
            ax.set_xticklabels(["Inside", "Count"])
            ax.set_title('Best Count Subnetwork Performance ' + model)
            ax.yaxis.grid(True)
            ax.annotate(" ".join(best_l0_stages), xy=(-0.2,.1))
            ax.annotate("L0 Init: " + str(best_l0_init), xy=(.5, .2))

            # Save the figure and show
            plt.tight_layout()
            plt.savefig(model + ' Best_Count_Subnetwork_Performance.png')
            plt.show()

            # Build the plot
            fig, ax = plt.subplots()
            ax.bar(range(2), [best_inside_ablate, best_count_ablate], align='center', alpha=0.5, ecolor='black', capsize=10)
            ax.set_ylabel('Accuracy')
            ax.set_xticks(range(2))
            ax.set_xticklabels(["Inside", "Count"])
            ax.set_title('Best Count Subnetwork Ablated Performance ' + model)
            ax.yaxis.grid(True)
            ax.annotate(" ".join(best_l0_stages), xy=(-0.2,.1))
            ax.annotate("L0 Init: " + str(best_l0_init), xy=(.5, .2))

            # Save the figure and show
            plt.tight_layout()
            plt.savefig(model + ' Best_Count_Subnetwork_Ablated_Performance.png')
            plt.show()

            dfs = []