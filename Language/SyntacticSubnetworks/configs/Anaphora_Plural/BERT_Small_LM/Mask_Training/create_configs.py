import yaml
import os
import copy
import pandas as pd

with open("Base.yaml", 'r') as stream:
    base = yaml.load(stream, Loader=yaml.FullLoader)

summary = pd.read_csv("../../../../../Results/Anaphora_Plural/BERT_Small_LM/Mask_Parameters/summary.csv")

os.makedirs("Mask_Configs", exist_ok=True)

configs = summary.groupby("0_Model_ID") # This gives us one group for each config to produce

for _, grp in configs:
    model_id = grp["Base Model"].iloc[0]
    task = grp["0_train_task"].iloc[0].item()
    lr = grp["3_lr"].iloc[0].item()
    l0_init = grp["3_l0_init"].iloc[0].item()
    stages = grp["3_l0_stages"].iloc[0]

    config = copy.deepcopy(base)
    config_name = model_id + "_T" + str(task)
    config["exp_dir"] = os.path.join(config["exp_dir"], config_name)
    config["results_dir"] = os.path.join(config["results_dir"], config_name)
    config["pretrained_weights"]["backbone"] = os.path.join(config["pretrained_weights"]["backbone"], model_id + "_backbone.pt")
    config["lr_list"] = [lr]
    config["l0_init_list"] = [l0_init]
    config["l0_stage_list"] = [int(stages)]
    config["train_task"] = str(task)

    config_path = os.path.join("Mask_Configs", config_name + ".yaml")
    with open(config_path, 'w') as cfg_file:
        yaml.dump(config, cfg_file)
