import yaml
import os
import copy
import pandas as pd

os.makedirs("Random_Configs", exist_ok=True)

mask_configs = os.listdir("Mask_Configs")

for mask_config in mask_configs:
    with open(os.path.join("Mask_Configs", mask_config), 'r') as stream:
        base = yaml.load(stream, Loader=yaml.FullLoader)
    
        random_config = copy.deepcopy(base)
        random_config["exp_dir"] = random_config["exp_dir"].replace("Mask_Training", "Random_Training")
        random_config["results_dir"] = random_config["results_dir"].replace("Mask_Training", "Random_Training")
        random_config["pretrained_weights"]["backbone"] = False
        random_config["save_models"] = False


        config_path = os.path.join("Random_Configs", mask_config)
        with open(config_path, 'w') as cfg_file:
            yaml.dump(random_config, cfg_file)