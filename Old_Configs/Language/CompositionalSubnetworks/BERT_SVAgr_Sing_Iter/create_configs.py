import yaml
import os
import copy

with open("BERT_base_config.yaml", 'r') as stream:
    base = yaml.load(stream, Loader=yaml.FullLoader)

os.makedirs("Experiment_Configs", exist_ok=True)

train_tasks = ['1', '2'] # Train Subject and Verb subnets

pretrained_models_dir = "/users/mlepori/data/mlepori/projects/Compositional_Subnetworks/Language/Models/BERT_Small/SVAgr_S_Models"
pretrained_models = ["1", "2", "3"]

l0_stages_list = [0, 2, 3]

for pretrained_model in pretrained_models:
    for train_task in train_tasks:
        for stages in l0_stages_list:
            # Create a new config file from base
            config = copy.deepcopy(base)
            
            # Edit individual parameters
            exp_name = "_".join(["BERT_Small", "model" + pretrained_model, "T" + train_task, str(stages)+"startStage"])
            config["exp_dir"] += exp_name
            config["results_dir"] += exp_name

            config["train_task"] = train_task

            config["pretrained_weights"]["backbone"] = os.path.join(pretrained_models_dir, pretrained_model + "_backbone.pt")

            config["l0_stages"] = stages

            # Save files
            config_path = os.path.join("Experiment_Configs", exp_name + ".yaml")
            with open(config_path, 'w') as cfg_file:
                yaml.dump(config, cfg_file)