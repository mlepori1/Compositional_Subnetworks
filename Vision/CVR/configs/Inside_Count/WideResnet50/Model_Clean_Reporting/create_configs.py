import yaml
import os
import copy

with open("WideResnet_Inside_Count_Base.yaml", 'r') as stream:
    base = yaml.load(stream, Loader=yaml.FullLoader)

os.makedirs("Configs", exist_ok=True)


pretrained_models_dir = "/users/mlepori/data/mlepori/projects/Compositional_Subnetworks/Vision/Models/Inside_Count/WideResnet50/Model_Training"
pretrained_models = ["aa1753cd-4f3e-406e-9e6f-f98e929d05c5", "a6914c5a-fa03-47d5-b3be-a51ba1d2b6fa", "c93c9e01-3235-4d48-8a6a-8de6260c46b6"]

for pretrained_model in pretrained_models:
    # Create a new config file from base
    config = copy.deepcopy(base)
    
    # Edit individual parameters
    config["exp_dir"] = os.path.join(config["exp_dir"], pretrained_model)
    config["results_dir"] = os.path.join(config["results_dir"], pretrained_model)

    config["pretrained_weights"]["backbone"] = os.path.join(pretrained_models_dir, pretrained_model + "_backbone.pt")
    config["pretrained_weights"]["mlp"] = os.path.join(pretrained_models_dir, pretrained_model + "_mlp.pt")
    config["pretrained_weights"]["embedding"] = False

    # Save files
    config_path = os.path.join("Configs", pretrained_model + ".yaml")
    with open(config_path, 'w') as cfg_file:
        yaml.dump(config, cfg_file)