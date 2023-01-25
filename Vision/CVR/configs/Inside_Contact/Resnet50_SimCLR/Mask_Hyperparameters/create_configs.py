import yaml
import os
import copy

with open("Base.yaml", 'r') as stream:
    base = yaml.load(stream, Loader=yaml.FullLoader)

os.makedirs("Mask_Configs", exist_ok=True)

l0_inits = [0.1, 0.05, 0.0, -0.05] # Search
train_tasks = ['104', '105'] 

pretrained_models_dir = "/users/mlepori/data/mlepori/projects/Compositional_Subnetworks/Vision/Models/Inside_Contact/Resnet50_SimCLR/Model_Training"
pretrained_models = ["42ef1a92-988d-436b-a9cd-e4ff7bb9d7fe", "081d4fc7-4ba4-401e-8263-48a5bb51c1a2", "e378984c-db6a-423f-8b71-7a06bbd0cfc6"]

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