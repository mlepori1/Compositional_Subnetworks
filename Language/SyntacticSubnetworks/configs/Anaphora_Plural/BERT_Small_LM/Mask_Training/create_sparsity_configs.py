import yaml
import os
import copy
import pandas as pd


os.makedirs("Sparsity_Configs", exist_ok=True)
configs = os.listdir("Mask_Configs")

for config in configs:
    config_name = config[:-5]
    stream = open(os.path.join("./Mask_Configs", config), 'r')
    config = yaml.load(stream, Loader=yaml.FullLoader)
    stream.close()
    sparse_config = copy.deepcopy(config)
    sparse_config["ablation_strategies"] = ["none", "zero"]
    sparse_config["seed_list"] = [0]
    sparse_config["test_tasks"] = ['1']
    sparse_config["max_epochs"] = 0
    sparse_config["save_models"] = False
    sparse_config["train_masks"]["backbone"] = False
    sparse_config["train_masks"]["mlp"] = False
    pt_root_dir = sparse_config["exp_dir"]
    sparse_config["exp_dir"] = pt_root_dir.replace("Mask_Training", "L0_Reporting")

    res_dir = sparse_config["results_dir"]

    backbones = [dir for dir in os.listdir(pt_root_dir) if "_backbone.pt" in dir]
    for b in backbones:
        sparse_config["pretrained_weights"]["backbone"] = os.path.join(pt_root_dir, b)
        sparse_config["results_dir"] = os.path.join(res_dir.replace("Mask_Training", "L0_Reporting"), b)

        config_path = os.path.join("Sparsity_Configs", b + ".yaml")
        with open(config_path, 'w') as cfg_file:
            yaml.dump(sparse_config, cfg_file)
