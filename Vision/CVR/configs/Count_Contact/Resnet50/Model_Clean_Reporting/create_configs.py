import yaml
import os
import copy

with open("Resnet_Count_Contact_Base.yaml", 'r') as stream:
    base = yaml.load(stream, Loader=yaml.FullLoader)

os.makedirs("Configs", exist_ok=True)


pretrained_models_dir = "/users/mlepori/data/mlepori/projects/Compositional_Subnetworks/Vision/Models/Count_Contact/Resnet50/Model_Training"
pretrained_models = ["49995cc4-ab6b-47f3-99a4-9c2de616c7e9", "732bcf3e-974c-44b6-ae4d-d1402d5961e4", "ae475d3e-3c13-40dd-8026-eec90d480feb"]

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