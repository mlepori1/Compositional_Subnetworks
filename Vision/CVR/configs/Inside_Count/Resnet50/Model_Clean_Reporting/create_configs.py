import yaml
import os
import copy

with open("Resnet_Inside_Count_Base.yaml", 'r') as stream:
    base = yaml.load(stream, Loader=yaml.FullLoader)

os.makedirs("Configs", exist_ok=True)


pretrained_models_dir = "/users/mlepori/data/mlepori/projects/Compositional_Subnetworks/Vision/Models/Inside_Count/Resnet50/Model_Training"
pretrained_models = ["0432d0b8-1719-478c-826c-5b18771f6b91", "9d495246-75ef-4b88-b382-adcf8c9f2087", "883cd15e-cf44-43bd-8fe9-56a91090aef7"]

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