import yaml
import os
import copy

with open("Base.yaml", 'r') as stream:
    base = yaml.load(stream, Loader=yaml.FullLoader)

os.makedirs("Mask_Configs", exist_ok=True)

l0_inits = [0.1, 0.05, 0.0, -0.05] # Search
train_tasks = ['110', '111'] 

pretrained_models_dir = "/users/mlepori/data/mlepori/projects/Compositional_Subnetworks/Vision/Models/Inside_Count/WideResnet50/Model_Training"
pretrained_models = ["aa1753cd-4f3e-406e-9e6f-f98e929d05c5", "a6914c5a-fa03-47d5-b3be-a51ba1d2b6fa", "c93c9e01-3235-4d48-8a6a-8de6260c46b6"]

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
                config_path = os.path.join("Mask_Configs", pretrained_model + "_" + exp_name + ".yaml")
                with open(config_path, 'w') as cfg_file:
                    yaml.dump(config, cfg_file)