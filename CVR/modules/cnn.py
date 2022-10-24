
import os
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
from torch import nn as nn
from torch.nn import functional as F

from torchvision import models

from models.vits import vit_small as vit_small_moco

from models.scn import SCL
from models.wren import WReN
from models.resnet18 import ResNet, L0Conv2d
from models.lenet import LeNet
from models.vgg11 import VGG11
#from models.mlpEncoder import L0MLP, MLP
from models.decisionMLP import MLP, L0MLP, L0UnstructuredLinear

class Base(pl.LightningModule):

    def load_finetune_weights(self, checkpoint):
        print("*"*10 + "load finetune weights ...")
        # CNN.load_fron
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

    def l0_loss(self, y_hat, y):
        error_loss = F.cross_entropy(y_hat, y)
        l0_loss = 0.0
        masks = []
        if self.train_masks["mlp"]:
            for layer in self.mlp.modules():
                if hasattr(layer, "mask"):
                    masks.append(layer.mask)
        if self.train_masks["backbone"]:
            for layer in self.backbone.modules():
                if hasattr(layer, "mask"):
                    masks.append(layer.mask)
        l0_loss = sum(m.sum() for m in masks)
        return (error_loss + (self.lamb * l0_loss), l0_loss)  
      
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

        results = {f"test_{k}": v for k, v in logs.items()}
        return results

    def test_epoch_end(self, outputs):

        keys = list(outputs[0].keys())
        results = {k: torch.cat([x[k] for x in outputs]).cpu().numpy() for k in keys}
        self.test_results = results

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.wd)



class CNN(Base):

    def __init__(
        self,
        backbone: str ='resnet50',
        lr: float = 1e-4,
        wd: float = 1e-4,
        n_tasks: int = 103,
        mlp_dim: int = 128,
        mlp_hidden_dim: int = 2048,
        task_embedding: int = 0, #64
        ssl_pretrain: bool = False,
        **kwargs
    ):
        """
        Args:
            input_height: height of the images
            kl_coeff: coefficient for kl term of the loss
            lr: learning rate for Adam
        """
        self.save_hyperparameters()

        super(CNN, self).__init__()

        feature_extract = True
        num_classes = 1
        use_pretrained = False
        self.mlp_hidden_dim = mlp_hidden_dim
        self.mlp_out_dim = mlp_dim
        self.l0_components = kwargs["l0_components"] # High level, are the backbone and mlp l0
        self.train_masks = kwargs["train_masks"]
        self.train_weights = kwargs["train_weights"]
        self.pretrained_weights = kwargs["pretrained_weights"]
        self.eval_only = kwargs["eval_only"]
        
        if "ablate_mask" in kwargs:
            self.ablate_mask = kwargs["ablate_mask"]
        else:
            self.ablate_mask = None

        if "l0_stages" in kwargs:
            self.l0_stages = kwargs["l0_stages"] # Granular, what parts of the backbone are L0
        else:
            self.l0_stages = None

        if backbone == "resnet18":
            """ Resnet18
            """
            assert(self.l0_components["backbone"] == False)
            self.backbone = ResNet(isL0=False, mask_init_value=1, embed_dim=1024)
            num_ftrs = self.backbone.embed_dim

            # If the pretrained weights path is not None, then load up pretrained weights!
            if self.pretrained_weights["backbone"] != False:
                self.backbone.load_state_dict(torch.load(self.pretrained_weights["backbone"]), strict=False)

            # If not training the weights in the backbone, freeze it
            if self.train_weights["backbone"] == False:
                self.backbone.train(False)

                for layer in self.backbone.modules():
                    if type(layer) == L0Conv2d or type(layer) == nn.BatchNorm2d:
                        layer.weight.requires_grad = False
                        if layer.bias != None: 
                            layer.bias.requires_grad = False


        elif backbone == "L0resnet18":
            """ L0Resnet18
            """

            assert(self.l0_components["backbone"] == True)

            # Get L0 parameters
            l0_init = kwargs["l0_init"]
            self.lamb = kwargs["l0_lambda"]
            
            # Define backbone structure
            self.backbone = ResNet(isL0=True, mask_init_value=l0_init, embed_dim=1024, ablate_mask=self.ablate_mask, l0_stages=self.l0_stages) # Defines the structure of L0 Resnet

            if self.pretrained_weights["backbone"] != False:
                self.backbone.load_state_dict(torch.load(self.pretrained_weights["backbone"]), strict=False)

            # If you don't want to train the L0 backbone mask, freeze the mask
            if self.train_masks["backbone"] == False:
                for layer in self.backbone.modules():
                    if type(layer) == L0Conv2d: # freeze conv layers
                        if not self.train_masks and layer.l0 == True: # Only decision is whether to freeze mask
                            layer.mask_weight.requires_grad = False
            
            # If you don't want to train the backbone weights, freeze em
            if self.train_weights["backbone"] == False:
                for layer in self.backbone.modules():
                    if type(layer) == L0Conv2d or type(layer) == nn.BatchNorm2d: # freeze all weights and biases
                        layer.weight.requires_grad = False
                        if layer.bias != None: 
                            layer.bias.requires_grad = False
            
            if self.train_masks["backbone"] == False and self.train_weights["backbone"] == False:
                self.backbone.train(False)

            num_ftrs = self.backbone.embed_dim

        if backbone == "vgg11":
            """ VGG16 backbone
            """
            assert(self.l0_components["backbone"] == False)
            self.backbone = VGG11(isL0=False, mask_init_value=1, embed_dim=8192)
            num_ftrs = self.backbone.embed_dim

            # If the pretrained weights path is not None, then load up pretrained weights!
            if self.pretrained_weights["backbone"] != False:
                self.backbone.load_state_dict(torch.load(self.pretrained_weights["backbone"]), strict=False)

            # If not training the weights in the backbone, freeze it
            if self.train_weights["backbone"] == False:
                self.backbone.train(False)

                for layer in self.backbone.modules():
                    if type(layer) == L0Conv2d or type(layer) == nn.BatchNorm2d:
                        layer.weight.requires_grad = False
                        if layer.bias != None: 
                            layer.bias.requires_grad = False


        elif backbone == "L0vgg11":
            """ L0VGG16 backbone
            """

            assert(self.l0_components["backbone"] == True)

            # Get L0 parameters
            l0_init = kwargs["l0_init"]
            self.lamb = kwargs["l0_lambda"]

            # Define backbone structure
            self.backbone = VGG11(isL0=True, mask_init_value=l0_init, embed_dim=8192, ablate_mask=self.ablate_mask) # Defines the structure of L0 VGG16

            if self.pretrained_weights["backbone"] != False:
                self.backbone.load_state_dict(torch.load(self.pretrained_weights["backbone"]), strict=False)

            # If you don't want to train the L0 backbone mask, freeze the mask
            if self.train_masks["backbone"] == False:
                for layer in self.backbone.modules():
                    if type(layer) == L0Conv2d: # freeze conv layers
                        if not self.train_masks and layer.l0 == True: # Only decision is whether to freeze mask
                            layer.mask_weight.requires_grad = False
            
            # If you don't want to train the backbone weights, freeze em
            if self.train_weights["backbone"] == False:
                for layer in self.backbone.modules():
                    if type(layer) == L0Conv2d or type(layer) == nn.BatchNorm2d: # freeze all weights and biases
                        layer.weight.requires_grad = False
                        if layer.bias != None: 
                            layer.bias.requires_grad = False
            
            if self.train_masks["backbone"] == False and self.train_weights["backbone"] == False:
                self.backbone.train(False)

            num_ftrs = self.backbone.embed_dim

        elif backbone == "resnet50":
            """ Resnet50
            """
            self.backbone = models.resnet50(pretrained=use_pretrained, progress=False, num_classes=num_classes)
            num_ftrs = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()

        if backbone == "lenet":
            """ LeNet
            """
            assert(self.l0_components["backbone"] == False)
            self.backbone = LeNet(isL0=False, mask_init_value=1, embed_dim=7680)
            num_ftrs = self.backbone.embed_dim

            # If the pretrained weights path is not None, then load up pretrained weights!
            if self.pretrained_weights["backbone"] != False:
                print("Loading LeNet Backbone weights...")
                self.backbone.load_state_dict(torch.load(self.pretrained_weights["backbone"]), strict=False)

            # If not training the weights in the backbone, freeze it
            if self.train_weights["backbone"] == False:
                print("Freezing LeNet Backbone weights...")
                self.backbone.train(False)

                for layer in self.backbone.modules():
                    if type(layer) == L0Conv2d or type(layer) == nn.BatchNorm2d:
                        layer.weight.requires_grad = False
                        if layer.bias != None: 
                            layer.bias.requires_grad = False

        elif backbone == "L0lenet":
            """ L0 LeNet
            """
            assert(self.l0_components["backbone"] == True)
            # Get L0 parameters
            l0_init = kwargs["l0_init"]
            self.lamb = kwargs["l0_lambda"]

            # Define backbone structure
            self.backbone = LeNet(isL0=True, mask_init_value=l0_init, embed_dim=7680) # Defines the structure of L0 LeNet

            if self.pretrained_weights["backbone"] != None:
                print("Loading L0 LeNet Backbone weights...")
                self.backbone.load_state_dict(torch.load(self.pretrained_weights["backbone"]), strict=False)

            # If you don't want to train the L0 backbone mask, freeze the mask
            if self.train_masks["backbone"] == False:
                print("Freezing L0 LeNet Mask Weights...")
                for layer in self.backbone.modules():
                    if type(layer) == L0Conv2d: # freeze conv layers
                        if not self.train_mask and layer.l0 == True: # Only decision is whether to freeze mask
                            layer.mask_weight.requires_grad = False
            
            # If you don't want to train the backbone weights, freeze em
            if self.train_weights["backbone"] == False:
                print("Freezing L0 LeNet weights...")
                for layer in self.backbone.modules():
                    if type(layer) == L0Conv2d: # freeze all weights and biases
                        layer.weight.requires_grad = False
                        if layer.bias != None: 
                            layer.bias.requires_grad = False
            
            if self.train_masks["backbone"] == False and self.train_weights["backbone"] == False:
                self.backbone.train(False)

            num_ftrs = self.backbone.embed_dim

        elif backbone == "vit_small":
            self.backbone = vit_small_moco(img_size=128, stop_grad_conv1=True)
            self.backbone.head = nn.Identity()
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
        self.mlp = MLP(num_ftrs + task_embedding, self.mlp_hidden_dim, self.mlp_out_dim)

        if self.l0_components["mlp"]:
            print("Constructing L0 MLP...")
            self.mlp = L0MLP(self.mlp, kwargs["l0_init"], ablate_mask=self.ablate_mask)
            self.lamb = kwargs["l0_lambda"]


        if self.pretrained_weights["mlp"]:
            print("Loading MLP weights...")
            self.mlp.load_state_dict(torch.load(self.pretrained_weights["mlp"]), strict=False)

        if not self.train_weights["mlp"]:
            # Freeze weights except for mask weight
            print("Freezing MLP weights...")
            for layer in self.mlp.model.children():
                if isinstance(layer, L0UnstructuredLinear) or isinstance(layer, nn.Linear):
                    layer.bias.requires_grad = False
                    layer.weight.requires_grad = False

        if not self.train_masks["mlp"]:
            # Freeze mask weight
            print("Freezing MLP mask weights...")
            for layer in self.mlp.model.children():
                if isinstance(layer, L0UnstructuredLinear):
                    layer.mask_weight.requires_grad = False

    def init_networks(self):
        # define encoder, decoder, fc_mu and fc_var
        pass

    def forward(self, x, task_idx=None):

        x_size = x.shape
        x = x.reshape([x_size[0]*4, x_size[2], x_size[3], x_size[4]]) # Unpack each problem. [N problems, 4 images, rgb, height, width] -> [N*4 images, rgb, height, width]

        x = self.backbone(x) # Get representation for each image
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

class SCN(Base):

    def __init__(
        self,
        backbone: str ='scl',
        lr: float = 5e-3,
        wd: float = 1e-2,
        n_tasks: int = 103,
        mlp_dim: int = 128,
        mlp_hidden_dim: int = 2048,
        task_embedding: int = 0, #64
        ssl_pretrain: bool = False,
        # task_embedding:
        **kwargs
    ):
        """
        Args:
            input_height: height of the images
            kl_coeff: coefficient for kl term of the loss
            lr: learning rate for Adam
        """
        self.save_hyperparameters()

        super(SCN, self).__init__()

        self.hidden_size = mlp_dim

        if task_embedding>0:
            self.task_embedding = nn.Embedding(n_tasks, task_embedding)
        else:
            self.task_embedding = None

        self.backbone = SCL(
            image_size=128,
            set_size=5,
            conv_channels=[3, 16, 16, 32, 32, 32],
            conv_output_dim=80,
            attr_heads=10,
            attr_net_hidden_dims=[128],
            rel_heads=80,
            rel_net_hidden_dims=[64, 23, 5],
            task_emb_size=task_embedding,
        )


    def forward(self, x, task_idx=None):


        x_task = self.task_embedding(task_idx) if task_idx is not None else None

        out = self.backbone(x, x_task)

        return out

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument("--backbone", type=str, default='resnet50')
        parser.add_argument("--wd", type=float, default=5e-3)
        parser.add_argument("--lr", type=float, default=1e-2)

        parser.add_argument("--n_tasks", type=int, default=103)
        parser.add_argument("--task_embedding", type=int, default=0)

        return parser


class WREN(Base):

    def __init__(
        self,
        backbone: str ='wren',
        lr: float = 1e-4,
        wd: float = 0,
        n_tasks: int = 103,
        mlp_dim: int = 128,
        mlp_hidden_dim: int = 2048,
        task_embedding: int = 0, #64
        ssl_pretrain: bool = False,
        **kwargs
    ):
        """
        Args:
            input_height: height of the images
            kl_coeff: coefficient for kl term of the loss
            lr: learning rate for Adam
        """
        self.save_hyperparameters()

        super(WREN, self).__init__()

        feature_extract = True
        num_classes = 1
        use_pretrained = False

        self.hidden_size = mlp_dim

        if task_embedding>0:
            self.task_embedding = nn.Embedding(n_tasks, task_embedding)
        else:
            self.task_embedding = None

        self.backbone = WReN(task_emb_size=task_embedding)

    def forward(self, x, task_idx=None):

        x_task = self.task_embedding(task_idx) if task_idx is not None else None

        out = self.backbone(x, x_task)

        return out

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument("--backbone", type=str, default='wren')
        parser.add_argument("--wd", type=float, default=1e-4)
        parser.add_argument("--lr", type=float, default=1e-4)

        parser.add_argument("--n_tasks", type=int, default=103)
        parser.add_argument("--task_embedding", type=int, default=0)

        return parser


class SCNHead(Base):

    def __init__(
        self,
        backbone: str ='resnet50',
        lr: float = 5e-3,
        wd: float = 1e-2,
        n_tasks: int = 103,
        mlp_dim: int = 128,
        mlp_hidden_dim: int = 2048,
        task_embedding: int = 0, #64
        ssl_pretrain: bool = False,
        **kwargs
    ):
        """
        Args:
            input_height: height of the images
            kl_coeff: coefficient for kl term of the loss
            lr: learning rate for Adam
        """
        self.save_hyperparameters()

        super(SCNHead, self).__init__()

        feature_extract = True
        num_classes = 1
        use_pretrained = False

        self.hidden_size = mlp_dim

        if backbone == "resnet18":
            """ Resnet18
            """
            self.backbone = models.resnet18(pretrained=use_pretrained, progress=False, num_classes=num_classes)
            num_ftrs = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()

        elif backbone == "resnet50":
            """ Resnet50
            """
            self.backbone = models.resnet50(pretrained=use_pretrained, progress=False, num_classes=num_classes)
            num_ftrs = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()


        if task_embedding>0:
            self.task_embedding = nn.Embedding(n_tasks, task_embedding)
            self.task_embedding_size = task_embedding
        else:
            self.task_embedding = None
            self.task_embedding_size = 0

        self.head = SCL(
            image_size=num_ftrs+task_embedding,
            set_size=5,
            conv_channels=[],
            conv_output_dim=mlp_hidden_dim,
            attr_heads=128,
            attr_net_hidden_dims=[256],
            rel_heads=mlp_hidden_dim,
            rel_net_hidden_dims=[64, 23, 5],
            task_emb_size=task_embedding,
        )

    def load_finetune_weights(self, checkpoint):
        print("*"*10 + "load finetune weights ...")
        model_temp = self.__class__.load_from_checkpoint(checkpoint)
        self.backbone.load_state_dict(model_temp.backbone.state_dict())
        self.head.load_state_dict(model_temp.head.state_dict())

    def load_backbone_weights(self, checkpoint):
        print("*"*10 + "load ckpt weights ...")
        self.backbone.load_state_dict(torch.load(checkpoint)['model'], strict=False)

    def forward(self, x, task_idx=None):

        x_task = self.task_embedding(task_idx) if task_idx is not None else None

        x_size = x.shape
        x = x.reshape([x_size[0]*4, x_size[2], x_size[3], x_size[4]])
        x = self.backbone(x)
        x = x.reshape([x_size[0], 4, -1])
        out = self.head(x, x_task)

        return out

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument("--backbone", type=str, default='resnet50')
        parser.add_argument("--wd", type=float, default=1e-4)
        parser.add_argument("--lr", type=float, default=1e-3)

        parser.add_argument("--n_tasks", type=int, default=103)
        parser.add_argument("--mlp_hidden_dim", type=int, default=256)
        parser.add_argument("--task_embedding", type=int, default=0)

        # parser.add_argument("--ssl_pretrain", action='store_true')

        return parser
