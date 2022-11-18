import torch

t = torch.load("/users/mlepori/data/mlepori/projects/Compositional_Subnetworks/Models/resnet50/Contact_Inside_Experiments/test/1_mlp.pt_preweights")
print(t.sum())
t = torch.load("/users/mlepori/data/mlepori/projects/Compositional_Subnetworks/Models/resnet50/Contact_Inside_Experiments/test/1_mlp.pt_postweights")
print(t.sum())
t = torch.load("/users/mlepori/data/mlepori/projects/Compositional_Subnetworks/Models/resnet50/Contact_Inside_Experiments/test/1_mlp.pt_testweights")
print(t.sum())

