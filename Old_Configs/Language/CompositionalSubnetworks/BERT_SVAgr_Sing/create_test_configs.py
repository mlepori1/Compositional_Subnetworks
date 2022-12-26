import yaml
import os
import copy
import pandas as pd

best_configurations = pd.read_csv("../../../../Results/BERT_Small/SVAgr_S_Experiments/best_configurations.csv")

os.makedirs("Test_Configs", exist_ok=True)

for _, row in best_configurations.iterrows():
    exp_dir = row["0_exp_dir"]
    config = "Experiment_Configs/" + os.path.basename(os.path.normpath(exp_dir)) + ".yaml"
    with open(config, 'r') as stream:
        base = yaml.load(stream, Loader=yaml.FullLoader)

    lr = row["3_lr"]

    base["seed_list"] = [1, 2, 3]
    base["lr_list"] = [lr]
    base["evaluation_type"] = "test"
    base["save_models"] = True
    base["results_dir"] = base["results_dir"].replace("SVAgr_S_Experiments", "SVAgr_S_Tests")
    base["exp_dir"] = base["exp_dir"].replace("SVAgr_S_Experiments", "SVAgr_S_Tests")

    model_idx = base["exp_dir"].index("model") 
    model_name = base["exp_dir"][model_idx:model_idx+6]
    train_task = base["train_task"]

    exp_name = model_name + "_T" + train_task

    # Save files
    config_path = os.path.join("Test_Configs", exp_name + ".yaml")
    with open(config_path, 'w') as cfg_file:
        yaml.dump(base, cfg_file)