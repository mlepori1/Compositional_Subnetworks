import yaml
import os
import copy

with open("Base.yaml", 'r') as stream:
    base = yaml.load(stream, Loader=yaml.FullLoader)

os.makedirs("Mask_Configs", exist_ok=True)

l0_inits = [0.1, 0.05, 0.0, -0.05] # Search
train_tasks = ['11', '12'] # Train Subject and Verb subnets

pretrained_models_dir = "/users/mlepori/data/mlepori/projects/Compositional_Subnetworks/Language/Models/Anaphora_Singular/BERT_Small"
pretrained_models = ["065cac34-4260-463d-8f88-b492b12c0ce3", "d028ecec-e016-441a-be9c-3351f6f9d271", "8b9a5996-bf37-4b29-bf63-364295d9dda3"]

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