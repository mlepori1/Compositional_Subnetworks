
import os
from argparse import ArgumentError, ArgumentParser

import pytorch_lightning as pl
import torch
from torch import nn as nn
from torch.nn import functional as F

from models.resnet import *
from models.decisionMLP import MLP, L0Linear
from transformers import ViTModel, ViTConfig
import functools

class Base(pl.LightningModule):

    def load_finetune_weights(self, checkpoint):
        print("*"*10 + "load finetune weights ...")
        model_temp = self.__class__.load_from_checkpoint(checkpoint)
        # model.load_finetune_weights(model_temp)
        self.backbone.load_state_dict(model_temp.backbone.state_dict())

    def load_backbone_weights(self, checkpoint):
        print("*"*10 + "load ckpt weights ...")
        self.backbone.load_state_dict(torch.load(checkpoint)['model'], strict=False)


    def freeze_pretrained(self):
        for param in self.backbone.parameters():
            param.requires_grad = False


    def shared_step(self, batch):

        x, task_idx = batch # x = (Batch idx, 4 images per set, RGB, Height, Width)

        # creates artificial label
        x_size = x.shape
        perms = torch.stack([torch.randperm(4, device=self.device) for _ in range(x_size[0])], 0) # Permute examples within a problem (a grouping of 4 images pertaining to one rule), so the fourth image isn't always the odd on out
        y = perms.argmax(1) # In the original order, the fourth element was always the odd one out, so here, the argmax corresponds to
                            # 4 for each problem, corresponding to the out one out

        perms = perms + torch.arange(x_size[0], device=self.device)[:,None]*4 # Want to get the idx of each image, after flattening out the problems
        perms = perms.flatten()

        x = x.reshape([x_size[0]*4, x_size[2], x_size[3], x_size[4]])[perms].reshape([x_size[0], 4, x_size[2], x_size[3], x_size[4]]) # Permute the images within a problem, while keeping problems grouped together.
                                                                                                                                      # Out -> Batch (Number of problems), 4 images per problem, RGB, height, width

        if self.task_embedding:
            y_hat = self(x, task_idx) # This is the Forward() call for each model
        else:
            y_hat = self(x)

        return y_hat, y

    def l0_loss(self, y_hat, y, test_mode=False):
        if test_mode:
            error_loss = F.cross_entropy(y_hat, y, reduction="none")
        else:
            error_loss = F.cross_entropy(y_hat, y)

        l0_loss = 0.0
        masks = []
        if self.train_masks["mlp"]:
            for layer in self.mlp.modules():
                if hasattr(layer, "mask_weight"):
                    masks.append(layer.mask)
        if self.train_masks["backbone"]:
            for layer in self.backbone.modules():
                if hasattr(layer, "mask_weight"):
                    masks.append(layer.mask)
        l0_loss = sum(m.sum() for m in masks)
        return (error_loss + (self.lamb * l0_loss), l0_loss)  
      

    def get_l0_norm(self):
        masks = []
        for layer in self.mlp.modules():
                if hasattr(layer, "mask_weight"):
                    masks.append(layer.mask)
        for layer in self.backbone.modules():
            if hasattr(layer, "mask_weight"):
                masks.append(layer.mask)
        l0_norm = sum(m.sum() for m in masks)
        l0_max = torch.Tensor([sum([len(m.reshape(-1)) for m in masks])])
        return l0_norm, l0_max

    def step(self, batch, batch_idx):

        y_hat, y = self.shared_step(batch)

        if self.train_masks["backbone"] or self.train_masks["mlp"]:
            loss, l0_loss = self.l0_loss(y_hat, y)
        else:
            loss = F.cross_entropy(y_hat, y)
            l0_loss = torch.zeros(loss.shape)

        acc = torch.sum((y == torch.argmax(y_hat, dim=1))).float() / len(y)

        logs = {
            "loss": loss.reshape(1),
            "acc": acc.reshape(1),
            "L0": l0_loss.reshape(1)
        }

        return loss, logs

    def training_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"metrics/train_{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"metrics/val_{k}": v for k, v in logs.items()})
        return loss

    def test_step(self, batch, batch_idx):

        y_hat, y = self.shared_step(batch)

        if self.train_masks["backbone"] or self.train_masks["mlp"]:
            loss, l0_norm = self.l0_loss(y_hat, y)
        else:
            loss = F.cross_entropy(y_hat, y, reduction="none")
            if self.l0_components["backbone"] or self.l0_components["mlp"]:
                l0_norm, l0_max = self.get_l0_norm()
            else:
                l0_norm = torch.Tensor([0])
                l0_max = torch.Tensor([0])


        #acc = torch.sum((y == torch.argmax(y_hat, dim=1))).float() / len(y)
        acc = (y == torch.argmax(y_hat, dim=1)) * 1.

        logs = {
            "loss": loss.reshape(-1),
            "acc": acc.reshape(-1),
            "L0": l0_norm.reshape(1), # Always the same during test
            "L0_Max": l0_max.reshape(1) # Always the same during test
        }

        results = {f"test_{k}": v for k, v in logs.items()}
        return results

    def test_epoch_end(self, outputs):

        keys = list(outputs[0].keys())
        results = {k: torch.cat([x[k] for x in outputs]).cpu().numpy() for k in keys}
        self.test_results = results

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.wd)



class Model(Base):

    def __init__(
        self,
        backbone: str ='resnet50',
        lr: float = 1e-4,
        wd: float = 1e-4,
        n_tasks: int = 103,
        mlp_dim: int = 128,
        mlp_hidden_dim: int = 2048,
        task_embedding: int = 0, #64
        **kwargs
    ):

        self.save_hyperparameters()

        super(Model, self).__init__()

        self.mlp_hidden_dim = mlp_hidden_dim
        self.mlp_out_dim = mlp_dim
        self.l0_components = kwargs["l0_components"] # High level, are the backbone and mlp l0
        self.train_masks = kwargs["train_masks"]
        self.train_weights = kwargs["train_weights"]
        self.pretrained_weights = kwargs["pretrained_weights"]
        self.backbone_type = backbone
        
        if "ablate_mask" in kwargs:
            self.ablate_mask = kwargs["ablate_mask"]
        else:
            self.ablate_mask = None

        if "l0_stages" in kwargs:
            self.l0_stages = kwargs["l0_stages"] # Granular, what parts of the backbone are L0
        else:
            self.l0_stages = None

        # set up model
        if self.l0_components["backbone"] == False:
            # Note: Instance norm doesn't use affine transform or track running statistics
            if self.backbone_type == "resnet50": self.backbone = resnet50(isL0=False, embed_dim=2048, norm_layer=nn.InstanceNorm2d)
            elif self.backbone_type == "wideresnet50": self.backbone = wide_resnet50_2(isL0=False, embed_dim=2048, norm_layer=nn.InstanceNorm2d)
            elif self.backbone_type == "vit": 
                # Set up ViT config
                config = ViTConfig(num_hidden_layers=12, hidden_dropout_prob=0, attention_probs_dropout_prob=0, image_size=128)
                config.l0 = False
                config.l0_start = -1
                config.mask_init_value=-1
                config.ablate_mask=None

                self.backbone = ViTModel(config)
                self.backbone.embed_dim = config.hidden_size
            else: raise ArgumentError("backbone not recognized")

            num_ftrs = self.backbone.embed_dim

           # If the pretrained weights path is not None, then load up pretrained weights!
            if self.pretrained_weights["backbone"] != False:
                self.backbone.load_state_dict(torch.load(self.pretrained_weights["backbone"]), strict=False)

            # If not training the weights in the backbone, freeze it
            if self.train_weights["backbone"] == False:
                self.backbone.train(False)

                for layer in self.backbone.modules():
                    if hasattr(layer, "weight") and layer.weight != None:
                        layer.weight.requires_grad = False
                    if hasattr(layer, "bias") and layer.bias != None: 
                            layer.bias.requires_grad = False

        elif self.l0_components["backbone"] == True:

            # Get L0 parameters
            self.l0_init = kwargs["l0_init"]
            self.lamb = kwargs["l0_lambda"]

            if self.backbone_type == "resnet50": self.backbone = resnet50(isL0=True, embed_dim=2048, mask_init_value=self.l0_init, ablate_mask=self.ablate_mask, l0_stages=self.l0_stages, norm_layer=nn.InstanceNorm2d)
            elif self.backbone_type == "wideresnet50": self.backbone = wide_resnet50_2(isL0=True, embed_dim=2048, mask_init_value=self.l0_init, ablate_mask=self.ablate_mask, l0_stages=self.l0_stages, norm_layer=nn.InstanceNorm2d)
            elif self.backbone_type == "vit": 
                # Set up ViT config
                config = ViTConfig(num_hidden_layers=12, hidden_dropout_prob=0, attention_probs_dropout_prob=0, image_size=128)
                config.l0 = True
                config.l0_start = self.l0_stages
                config.mask_init_value=self.l0_init
                config.ablate_mask=self.ablate_mask

                self.backbone = ViTModel(config)
                self.backbone.embed_dim = config.hidden_size

            else: raise ArgumentError("backbone not recognized")
            
            if self.pretrained_weights["backbone"] != False:
                self.backbone.load_state_dict(torch.load(self.pretrained_weights["backbone"]), strict=False)

            # If you don't want to train the L0 backbone mask, freeze the mask
            if self.train_masks["backbone"] == False:
                for layer in self.backbone.modules():
                    if hasattr(layer,"l0") and layer.l0 == True: # freeze l0 layers
                        layer.mask_weight.requires_grad = False
            
            # If you don't want to train the backbone weights, freeze em
            if self.train_weights["backbone"] == False:
                for layer in self.backbone.modules():
                    if hasattr(layer, "weight") and layer.weight != None:
                        layer.weight.requires_grad = False
                    if hasattr(layer, "bias") and layer.bias != None: 
                            layer.bias.requires_grad = False
            
            if self.train_masks["backbone"] == False and self.train_weights["backbone"] == False:
                self.backbone.train(False)

            num_ftrs = self.backbone.embed_dim

        # Set up embeddings
        if task_embedding>0:
            self.task_embedding = nn.Embedding(n_tasks, task_embedding)
        else:
            self.task_embedding = None

        # If there are pretrained embedding weights, load them up
        if self.task_embedding != None and self.pretrained_weights["embedding"]:
            print("Loading embedding weights...")
            self.task_embedding.load_state_dict(torch.load(self.pretrained_weights["embedding"]), strict=False)

        if not self.train_weights["embedding"] and self.task_embedding != None:
            print("Freezing embeddding weights...")
            self.task_embedding.weight.requires_grad = False

        # Set up MLP
        if self.l0_components["mlp"]:
            print("Constructing L0 MLP...")
            self.mlp = MLP(num_ftrs + task_embedding, self.mlp_hidden_dim, self.mlp_out_dim, isL0=True, mask_init_value=kwargs["l0_init"],  ablate_mask=self.ablate_mask)
            self.lamb = kwargs["l0_lambda"]
        else:
            self.mlp = MLP(num_ftrs + task_embedding, self.mlp_hidden_dim, self.mlp_out_dim, isL0=False)

        if self.pretrained_weights["mlp"]:
            print("Loading MLP weights...")
            self.mlp.load_state_dict(torch.load(self.pretrained_weights["mlp"]), strict=False)

        if not self.train_weights["mlp"]:
            # Freeze weights except for mask weight
            print("Freezing MLP weights...")
            for layer in self.mlp.model.children():
                if hasattr(layer, "weight"):
                    layer.weight.requires_grad = False
                if hasattr(layer, "bias"):
                    layer.bias.requires_grad = False

        if not self.train_masks["mlp"]:
            # Freeze mask weight
            print("Freezing MLP mask weights...")
            for layer in self.mlp.model.children():
                if hasattr(layer, "mask_weight"):
                    layer.mask_weight.requires_grad = False

    def init_networks(self):
        # define encoder, decoder, fc_mu and fc_var
        pass

    def forward(self, x, task_idx=None):

        x_size = x.shape
        x = x.reshape([x_size[0]*4, x_size[2], x_size[3], x_size[4]]) # Unpack each problem. [N problems, 4 images, rgb, height, width] -> [N*4 images, rgb, height, width]
        x = self.backbone(x) # Get representation for each image
        if self.backbone_type == "vit":
            x = x.last_hidden_state[:, 0, :] # Get first token from last layer

        if task_idx is not None:
            x_task = self.task_embedding(task_idx.repeat_interleave(4)) # Repeat_interleave repeats tensor values N times [1, 2].repeat_interleave(2) = [1, 1, 2, 2]
                                                                        # Because images for a problem are still grouped together, this gives the correct task idx for every image in a problem,
                                                                        # and thus the right task embedding
            x = torch.cat([x, x_task], 1) # mlp input is image representation cat task embedding
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=1) # Normalize the resulting MLP vectors

        x = x.reshape([-1, 4, self.mlp_out_dim]) # Reshape into (# problems, 4 images per problem, mlp size

        x = (x[:,:,None,:] * x[:,None,:,:]).sum(3).sum(2) # None indexing unsqueezes another dimensions at that index
            # So you have (# problems, 4 ims per problem, 1, mlp representation) * (# problems, 1, 4 ims per problem, mlp representation)
            # Calculates the dot product of each mlp vector with every other vector (as well as itself) via broadcasting. Then sums the total
            # similarity
        x = -x # The odd one out's total similarity to every other vector should be LOWER than the other vector's, so negate the dot products, making argmax(x) = odd one out because it is the least negative
        return x

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument("--backbone", type=str, default='resnet50')
        parser.add_argument("--wd", type=float, default=1e-4)
        parser.add_argument("--lr", type=float, default=1e-4)

        parser.add_argument("--n_tasks", type=int, default=103)
        parser.add_argument("--mlp_dim", type=int, default=128)
        parser.add_argument("--mlp_hidden_dim", type=int, default=2048)
        parser.add_argument("--task_embedding", type=int, default=0)

        return parser
