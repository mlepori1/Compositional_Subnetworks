import yaml
import os
import copy

with open("Base.yaml", 'r') as stream:
    base = yaml.load(stream, Loader=yaml.FullLoader)

os.makedirs("Mask_Configs", exist_ok=True)

l0_inits = [0.1, 0.05, 0.0, -0.05] # Search
train_tasks = ['1', '2'] # Train Subject and Verb subnets

pretrained_models_dir = "/users/mlepori/data/mlepori/projects/Compositional_Subnetworks/Language/Models/SV_Agr_Singular/BERT_Small_LM"
pretrained_models = ["da4466e3-3b9d-4932-af62-a8d548906254", "5bb85b5d-264b-4a80-b54f-d8c072480327", "4bb7a145-9597-4a1d-9923-e91f2d44183a"]

l0_stages_lists = [0, 2, 3]

for pretrained_model in pretrained_models:
    for train_task in train_tasks:
        for l0_init in l0_inits:
            for stages_list in l0_stages_lists:
                # Create a new config file from base
                config = copy.deepcopy(base)
                
                # Edit individual parameters
                exp_name = "_".join(["T" + train_task, str(l0_init), str(stages_list)+"stage"])
                config["exp_dir"] = os.path.join(config["exp_dir"], pretrained_model, str(train_task), exp_name)
                config["results_dir"] = os.path.join(config["results_dir"], pretrained_model, str(train_task), exp_name)

                config["train_task"] = train_task

                config["pretrained_weights"]["backbone"] = os.path.join(pretrained_models_dir, "Model_Training", pretrained_model + "_backbone.pt")

                config["l0_init_list"] = [l0_init]
                config["l0_stage_list"] = [stages_list]

                # Save files
                config_path = os.path.join("Mask_Configs", pretrained_model + "_" + exp_name + ".yaml")
                with open(config_path, 'w') as cfg_file:
                    yaml.dump(config, cfg_file)