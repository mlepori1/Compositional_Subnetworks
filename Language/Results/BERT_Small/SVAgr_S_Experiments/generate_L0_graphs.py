import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import collections

dirs = os.listdir(".")

models = ["model1", "model2", "model3"]
tasks = ["T1", "T2"]

init2l0s = collections.defaultdict(list)
init2ablates = collections.defaultdict(list)
for model in models:
    for task in tasks:
        for l0_init in ["_-0.05_", "_0.0_", "_0.1_"]:
            results = None

            for d in dirs:
                if model in d and task in d and "0startStage" in d and l0_init in d:
                    results = pd.read_csv(d + "/results.csv")
    
            init2ablates[l0_init].append(results[(results["0_ablation"] == "zero") & (results["1_task"] == 3) & (results["3_lr"] == 0.0001)]["2_test_acc"].tolist()[0])
            init2ablates[l0_init].append(results[(results["0_ablation"] == "zero") & (results["1_task"] == 4) & (results["3_lr"] == 0.0001)]["2_test_acc"].tolist()[0])
            init2l0s[l0_init].append(results[pd.isna(results["0_ablation"]) & (results["3_lr"] == 0.0001)]["2_test_L0"].tolist()[1])


print(init2ablates)
print(init2l0s)

x_axis = [-0.05, 0.0, 0.1]

mean_l0s = [np.mean(init2l0s["_-0.05_"]), np.mean(init2l0s["_0.0_"]), np.mean(init2l0s["_0.1_"])]
std_l0s = [np.std(init2l0s["_-0.05_"]), np.std(init2l0s["_0.0_"]), np.std(init2l0s["_0.1_"])]


# Build the plot
plt.errorbar(x_axis, mean_l0s, yerr=std_l0s)
plt.ylabel('L0')
plt.xlabel('L0_inits')

plt.title("Overall L0s vs Inits at LR=0.0001")
# Save the figure and show
plt.tight_layout()
plt.savefig("L0_init_vs_L0.png")
plt.show()


mean_accs = [np.mean(init2ablates["_-0.05_"]), np.mean(init2ablates["_0.0_"]), np.mean(init2ablates["_0.1_"])]
std_accs = [np.std(init2ablates["_-0.05_"]), np.std(init2ablates["_0.0_"]), np.std(init2ablates["_0.1_"])]

plt.clf()

# Build the plot
plt.errorbar(x_axis, mean_accs, yerr=std_accs)
plt.ylabel('Ablation_Performance')
plt.xlabel('L0_inits')

plt.title("Overall Ablation Accs vs Inits at LR=0.0001")
# Save the figure and show
plt.tight_layout()
plt.savefig("L0_init_vs_Abl_Accs.png")
plt.show()