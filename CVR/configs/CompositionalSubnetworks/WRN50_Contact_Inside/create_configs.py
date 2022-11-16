import yaml
import os
import copy

with open("WRN50_base_config.yaml", 'r') as stream:
    base = yaml.load(stream, Loader=yaml.FullLoader)

os.makedirs("Experiment_Configs", exist_ok=True)

l0_inits = [0.0, -0.01, -0.03, -0.05] # from section 5.3 of continuous sparsification, got rid of -.02 to minimize combinatorics
train_tasks = ['104', '105'] # Train insideness and contact subnets

pretrained_models_dir = "/gpfs/data/epavlick/mlepori/projects/Compositional_Subnetworks/Models/wide_resnet50/ConIn_Models"
pretrained_models = ["1", "2", "3"]

l0_stages_lists = [
    ["first", "stage_1", "stage_2", "stage_3", "stage_4"], 
    ["stage_2", "stage_3", "stage_4"],
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
                exp_name = "_".join(["WRN50", "model" + pretrained_model, "T" + train_task, str(l0_init), str(number_l0_stages)+"stage"])
                config["exp_dir"] += exp_name
                config["results_dir"] += exp_name

                config["train_task"] = train_task

                config["pretrained_weights"]["backbone"] = os.path.join(pretrained_models_dir, pretrained_model + "_backbone.pt")
                config["pretrained_weights"]["mlp"] = os.path.join(pretrained_models_dir, pretrained_model + "_mlp.pt")
                config["pretrained_weights"]["embedding"] = False

                config["l0_init_list"] = [l0_init]

                config["l0_stage_list"] = [stages_list]

                # Save files
                config_path = os.path.join("Experiment_Configs", exp_name + ".yaml")
                with open(config_path, 'w') as cfg_file:
                    yaml.dump(config, cfg_file)