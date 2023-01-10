import yaml
import os
import copy

with open("WideResnet_Count_Contact_Base.yaml", 'r') as stream:
    base = yaml.load(stream, Loader=yaml.FullLoader)

os.makedirs("Configs", exist_ok=True)


pretrained_models_dir = "/users/mlepori/data/mlepori/projects/Compositional_Subnetworks/Vision/Models/Count_Contact/WideResnet50/Model_Training"
pretrained_models = ["40892c97-5848-46cc-9cca-878a47792f3a", "f4f21ca4-942b-4f6b-9166-26a9bec253a9", "389a5942-bab7-489e-b126-7d591795239f"]

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