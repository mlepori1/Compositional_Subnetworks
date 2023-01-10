import yaml
import os
import copy

with open("Base.yaml", 'r') as stream:
    base = yaml.load(stream, Loader=yaml.FullLoader)

os.makedirs("Mask_Configs", exist_ok=True)

l0_inits = [0.1, 0.05, 0.0, -0.05] # Search
train_tasks = ['16', '17'] # Train Subject and Verb subnets

pretrained_models_dir = "/users/mlepori/data/mlepori/projects/Compositional_Subnetworks/Language/Models/Anaphora_Plural/BERT_Small_LM"
pretrained_models = ["c2ede735-a80d-4d31-b3a9-b6502142a2e3", "a04b0e46-c394-46a4-8663-378dd8f89674", "0b37a9ed-4265-40d5-8fd4-41a0f4b2ff8d"]

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