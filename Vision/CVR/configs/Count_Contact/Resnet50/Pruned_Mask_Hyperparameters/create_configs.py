import yaml
import os
import copy

with open("Base.yaml", 'r') as stream:
    base = yaml.load(stream, Loader=yaml.FullLoader)

os.makedirs("Mask_Configs", exist_ok=True)

l0_inits = [0.1, 0.05, 0.0, -0.05] # Search
train_tasks = ['116', '117'] 

pretrained_models_dir = "/users/mlepori/data/mlepori/projects/Compositional_Subnetworks/Vision/Models/Count_Contact/Resnet50/Pruning_Training"
pretrained_models = ["732bcf3e-974c-44b6-ae4d-d1402d5961e4_T115/f21191fc-373a-48c0-ba1c-7e0297c965f3", "49995cc4-ab6b-47f3-99a4-9c2de616c7e9_T115/db89fc41-7d31-4f19-b28d-173cefad8e35", "ae475d3e-3c13-40dd-8026-eec90d480feb_T115/61f77af9-0def-4c1b-8ae8-bd1e74c0d112"]

l0_stages_lists = [
    ["first", "stage_1", "stage_2", "stage_3", "stage_4"], 
    ["stage_3", "stage_4"],
    ["stage_4"]
    ] # Define how to ablate network, if ablating stage 1, also ablate the first conv. Control combinatorics

for pretrained_model in pretrained_models:
    for train_task in train_tasks:
        for l0_init in l0_inits:
            for stages_list in l0_stages_lists:
                # Create a new config file from base
                config = copy.deepcopy(base)
                
                # Edit individual parameters
                number_l0_stages = len(stages_list)
                exp_name = "_".join(["T" + train_task, str(l0_init), str(number_l0_stages)+"stage"])
                config["exp_dir"] = os.path.join(config["exp_dir"], pretrained_model, str(train_task), exp_name)
                config["results_dir"] = os.path.join(config["results_dir"], pretrained_model, str(train_task), exp_name)

                config["train_task"] = train_task

                config["pretrained_weights"]["backbone"] = os.path.join(pretrained_models_dir, pretrained_model + "_backbone.pt")
                config["pretrained_weights"]["mlp"] = os.path.join(pretrained_models_dir, pretrained_model + "_mlp.pt")
                config["pretrained_weights"]["embedding"] = False

                config["l0_init_list"] = [l0_init]

                config["l0_stage_list"] = [stages_list]

                # Save files
                config_path = os.path.join("Mask_Configs", pretrained_model.split("/")[-1] + "_" + exp_name + ".yaml")
                with open(config_path, 'w') as cfg_file:
                    yaml.dump(config, cfg_file)