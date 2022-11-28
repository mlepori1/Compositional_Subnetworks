import pandas as pd
import os
import numpy as np

dirs = os.listdir(".")

models = ["model1", "model2", "model3"]
tasks = ["105"]
stages = ["1stage", "2stage", "3stage", "5stage"]

dfs = []
for model in models:
    for task in tasks:
        # Get all results dfs
        for d in dirs:
            if model in d and task in d:
                dfs.append(pd.read_csv(d + "/results.csv"))
        
        results = pd.concat(dfs)
        if task == "104":
            pass
            '''
            original_results = results
            results = results[results["0_ablation"] == "zero"]
            results = results[results["1_task"] >= 107]
            results["test_diffs"] = results["2_test_acc"].diff()
            results = results[results["1_task"] == 108]
            results = results[results["test_diffs"] > 0]
            best_result = results[results["test_diffs"] == np.max(results["test_diffs"])]
            print(model)
            print(task)
            print(best_result)
            print(best_result["0_exp_dir"].item())
            print(best_result["0_Model_ID"])
            print("####")
            '''
        elif task == "105":
            print(dir)
            original_results = results
            results = results[results["0_ablation"] == "zero"]
            results = results[results["1_task"] >= 107]
            results["test_diffs"] = results["2_test_acc"].diff()
            results = results[results["1_task"] == 108]
            results = results[results["test_diffs"] < 0]
            best_result = results[results["test_diffs"] == np.min(results["test_diffs"])]
            print(model)
            print(task)
            print(best_result)
            print(best_result["0_exp_dir"].item())
            print(best_result["0_Model_ID"])
            print("####")
            dfs = []