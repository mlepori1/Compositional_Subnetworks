import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

root_path = "../Results/BERT_Small/SVAgr_S_Experiments/"
root_dir = os.listdir(root_path)

models = ["model1", "model2", "model3"]
tasks = ["1", "2"]
stages = ["0startStage", "2startStage", "3startStage"]

task_expected_performance = {
    "1" : {
        "high": 3,
        "low": 4
    },
    "2" : {
        "high": 4,
        "low": 3
    }
}

best_configurations = []
for model in models:
    for task in tasks:
        # Get all results dfs

        dfs = []
        for d in root_dir:
            if model in d and "T" + task in d:
                dfs.append(pd.read_csv(root_path + d + "/results.csv"))
        
        results = pd.concat(dfs)

        # Create a unique identifier for every subnet
        results["subnet_id"] = results["0_exp_dir"] + results["0_Model_ID"].astype(str)

        # First, filter for validation accuracy above 90% threshold
        ids = results[results["2_val_acc"] > .9]["subnet_id"]
        candidate_results = results.loc[results['subnet_id'].isin(ids)]

        # Now, check compute score for zero-ablated tests
        candidate_ablations = candidate_results[candidate_results["0_ablation"] == "zero"]
        ablation_groups = candidate_ablations.groupby(by=["subnet_id"])

        best_score = -np.inf
        best_id = None

        for id, group in ablation_groups:
            subnet_id = id
            high_acc = group[group["1_task"] == task_expected_performance[task]["high"]]["2_test_acc"].iloc[0]
            low_acc = group[group["1_task"] == task_expected_performance[task]["low"]]["2_test_acc"].iloc[0]
            score = (high_acc - 1.0) + np.min([0.25 - low_acc, 0.0]) # Highest score possible is a 0.0, indicating perfect performance on one task and chance (or worse) on the other
            if score > best_score:
                best_score = score
                best_id = id


        print("\n$$")
        print(best_score)
        print(best_id)
        print("&&\n")
        best_configuration = results.loc[results['subnet_id'] == best_id]
        best_configuration = best_configuration.loc[best_configuration['0_train'] == 1]
        print(len(best_configuration))
        #print(best_configuration)
        best_configurations.append(best_configuration)

best_configurations = pd.concat(best_configurations, axis=0)
best_configurations.to_csv("../Results/BERT_Small/SVAgr_S_Experiments/best_configurations.csv", index=False)
