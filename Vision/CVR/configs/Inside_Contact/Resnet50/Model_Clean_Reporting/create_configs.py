import yaml
import os
import copy

with open("Resnet_Inside_Contact_Base.yaml", 'r') as stream:
    base = yaml.load(stream, Loader=yaml.FullLoader)

os.makedirs("Configs", exist_ok=True)


pretrained_models_dir = "/users/mlepori/data/mlepori/projects/Compositional_Subnetworks/Vision/Models/Inside_Contact/Resnet50/Model_Training"
pretrained_models = ["b80c24dc-266d-4c4d-9967-ee59215ca185", "2b96162d-4c11-4eaa-822c-c0075d8337fb", "1d366e8a-e3b4-4453-80ed-640f69fdb3af"]

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