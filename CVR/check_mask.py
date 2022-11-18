import torch

t = torch.load("/users/mlepori/data/mlepori/projects/Compositional_Subnetworks/Models/resnet50/Contact_Inside_Experiments/test/1_mlp.pt_mask")
print(t.sum())