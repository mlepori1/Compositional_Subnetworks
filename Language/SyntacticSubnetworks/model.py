
import os
from argparse import ArgumentError, ArgumentParser

import pytorch_lightning as pl
import torch
from torch import nn as nn
from torch.nn import functional as F

from transformers import BertModel, BertConfig, BertTokenizer

class Base(pl.LightningModule):


    def shared_step(self, batch):

        # batch = Batch of tensors with shape [batch idx, 2 (ipt ids, attn mask), 4 (sentences per problem), max seq len]

        x_ipts = batch[:, 0, :, :]
        x_attn = batch[:, 1, :, :]

        x_size = x_ipts.shape

        # creates artificial label
        perms = torch.stack([torch.randperm(4, device=self.device) for _ in range(x_size[0])], 0) # Permute examples within a problem (a grouping of 4 sentences pertaining to one rule), so the fourth sentence isn't always the odd on out
        y = perms.argmax(1) # In the original order, the fourth element was always the odd one out, so here, the argmax corresponds to
                            # 4 for each problem, corresponding to the out one out

        perms = perms + torch.arange(x_size[0], device=self.device)[:,None]*4 # Want to get the idx of each sentence, after flattening out the problems
        perms = perms.flatten()

        x_ipts = x_ipts.reshape([x_size[0]*4, x_size[2]])[perms].reshape([x_size[0], 4, x_size[2]]) # Permute the sentence encodings within a problem, while keeping problems grouped together.
                                                                                            # Out -> Batch (Number of problems), 4 sentences per problem, max seq length
        x_attn = x_attn.reshape([x_size[0]*4, x_size[2]])[perms].reshape([x_size[0], 4, x_size[2]]) # Permute the sentence encodings within a problem, while keeping problems grouped together.
                                                                                            # Out -> Batch (Number of problems), 4 sentences per problem, max seq length
        y_hat = self(x_ipts, x_attn)

        return y_hat, y

    def l0_loss(self, y_hat, y, test_mode=False):
        if test_mode:
            error_loss = F.cross_entropy(y_hat, y, reduction="none")
        else:
            error_loss = F.cross_entropy(y_hat, y)

        l0_loss = 0.0
        masks = []
        if self.train_masks["backbone"]:
            for layer in self.backbone.modules():
                if hasattr(layer, "mask_weight"):
                    masks.append(layer.mask)
        l0_loss = sum(m.sum() for m in masks)
        return (error_loss + (self.lamb * l0_loss), l0_loss)  
      

    def get_l0_norm(self):
        masks = []
        for layer in self.backbone.modules():
            if hasattr(layer, "mask_weight"):
                masks.append(layer.mask)
        l0_norm = sum((m.bool()).float().sum().item() for m in masks)
        l0_norm = torch.tensor(int(l0_norm))
        l0_max = torch.tensor(int(sum([len(m.reshape(-1)) for m in masks])))
        return l0_norm, l0_max

    def step(self, batch, batch_idx):

        y_hat, y = self.shared_step(batch)

        if self.train_masks["backbone"]:
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

        if self.train_masks["backbone"]:
            loss, l0_norm = self.l0_loss(y_hat, y)
        else:
            loss = F.cross_entropy(y_hat, y, reduction="none")
            if self.l0_components["backbone"]:
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



class BertClf(Base):

    def __init__(
        self,
        backbone: str ='BERT_small',
        lr: float = 1e-4,
        wd: float = 0,
        **kwargs
    ):

        self.save_hyperparameters()

        super(BertClf, self).__init__()

        self.l0_components = kwargs["l0_components"] # Just for consistency with CVR, only backbone
        self.train_masks = kwargs["train_masks"]
        self.train_weights = kwargs["train_weights"]
        self.pretrained_weights = kwargs["pretrained_weights"]
        self.LM_init = kwargs["LM_init"]
        self.freeze_until = kwargs["freeze_until"]
        
        if "ablate_mask" in kwargs:
            self.ablate_mask = kwargs["ablate_mask"]
        else:
            self.ablate_mask = None

        if "l0_stages" in kwargs:
            self.l0_stages = kwargs["l0_stages"] # Granular, what parts of the backbone are L0. 
                                                 # In this case, an integer determining which layer to start at
        else:
            self.l0_stages = None

        # set up model
        if self.l0_components["backbone"] == False:
            # Just adding in BERT small for now, can expand to other variants if needed
            if backbone == "BERT_small": 
                bertConfig = BertConfig.from_pretrained("prajjwal1/bert-small")
                bertConfig.l0 = False # added in for continuous sparsification
                bertConfig.hidden_dropout_prob = 0.0
                bertConfig.attention_probs_dropout_prob = 0.0
                self.backbone = BertModel(config=bertConfig)
                self.tokenizer = BertTokenizer.from_pretrained("prajjwal1/bert-small")

            else: raise ValueError("backbone not recognized")

           # If the pretrained weights path is not None, then load up pretrained weights!
            if self.pretrained_weights["backbone"] != False:
                if self.LM_init:
                    # If downloading weights off the internet to initialize model, need to grab state dict
                    state_dict = BertModel.from_pretrained(self.pretrained_weights["backbone"]).state_dict()
                    self.backbone.load_state_dict(state_dict, strict=False)
                else:
                    self.backbone.load_state_dict(torch.load(self.pretrained_weights["backbone"]), strict=False)


            # If not training the weights in the backbone, freeze it
            if self.train_weights["backbone"] == False:
                self.backbone.train(False)
                for layer in self.backbone.modules():
                    if hasattr(layer, "weight"):
                        if layer.weight != None:
                            layer.weight.requires_grad = False
                    if hasattr(layer, "bias"):
                        if layer.bias != None: 
                            layer.bias.requires_grad = False

            # If training the weights, but want to freeze early layers
            # 0 indicates just loading embeddings
            if self.train_weights["backbone"] == True and self.freeze_until != -1:

                if self.pretrained_weights["backbone"] == False: 
                    raise ArgumentError("Freezing network without loading weights, this will freeze a random initialization!")
                    
                if self.freeze_until == 0:
                    modules = [self.backbone.embeddings]
                else:
                    modules = [self.backbone.embeddings, *self.backbone.encoder.layer[:self.freeze_until]]
                for module in modules:
                    for param in module.parameters():
                        param.requires_grad = False

        elif self.l0_components["backbone"] == True:

            # Get L0 parameters
            self.l0_init = kwargs["l0_init"]
            self.lamb = kwargs["l0_lambda"]

            if backbone == "BERT_small": 
                bertConfig = BertConfig.from_pretrained("prajjwal1/bert-small")
                bertConfig.l0 = True # added in for continuous sparsification
                bertConfig.l0_start = self.l0_stages
                bertConfig.mask_init_value = self.l0_init
                bertConfig.ablate_mask = self.ablate_mask
                bertConfig.hidden_dropout_prob = 0.0
                bertConfig.attention_probs_dropout_prob = 0.0
                self.backbone = BertModel(config=bertConfig)
                self.tokenizer = BertTokenizer.from_pretrained("prajjwal1/bert-small")

            else: raise ArgumentError("backbone not recognized")
            
           # If the pretrained weights path is not None, then load up pretrained weights!
            if self.pretrained_weights["backbone"] != False:
                if self.LM_init:
                    # If downloading weights off the internet to initialize model, need to grab state dict
                    state_dict = BertModel.from_pretrained(self.pretrained_weights["backbone"]).state_dict()
                    self.backbone.load_state_dict(state_dict, strict=False)
                else:
                    self.backbone.load_state_dict(torch.load(self.pretrained_weights["backbone"]), strict=False)

            # If you don't want to train the L0 backbone mask, freeze the mask
            if self.train_masks["backbone"] == False:
                for layer in self.backbone.modules():
                    if hasattr(layer, "mask_weight"):
                        layer.mask_weight.requires_grad = False
            
            # If you don't want to train the backbone weights, freeze em
            if self.train_weights["backbone"] == False:
                for layer in self.backbone.modules():
                    if hasattr(layer, "weight"):
                        if layer.weight != None: # freeze all weights and biases
                            layer.weight.requires_grad = False
                    if hasattr(layer, "bias"):
                        if layer.bias != None: 
                            layer.bias.requires_grad = False

            # If training the weights, but want to freeze early layers
            # 0 indicates just loading embeddings
            if self.train_weights["backbone"] == True and self.freeze_until != -1:

                if self.pretrained_weights["backbone"] == False: 
                    raise ArgumentError("Freezing network without loading weights, this will freeze a random initialization!")
                    
                if self.freeze_until == 0:
                    modules = [self.backbone.embeddings]
                else:
                    modules = [self.backbone.embeddings, *self.backbone.encoder.layer[:self.freeze_until]]
                for module in modules:
                    for param in module.parameters():
                        param.requires_grad = False
            
            if self.train_masks["backbone"] == False and self.train_weights["backbone"] == False:
                self.backbone.train(False)


    def prune(self):
        self.backbone.train(False) # For computing hard masks
        for layer in self.backbone.modules():
            if hasattr(layer, "mask_weight"):
                mask = layer.compute_mask()   
                layer.weight.data = layer.weight * mask # Prune
                layer.init_mask()  # Reinitialize mask

        self.backbone.train(True) # Ready to continue training


    def init_networks(self):
        pass

    def forward(self, x_ipt, x_attn):

        x_size = x_ipt.shape
        x_ipt = x_ipt.reshape([x_size[0]*4, x_size[2]]) # Unpack each problem. [N problems, 4 sentences, maxseqlen tokens] -> [N*4 sentences, maxseqlen tokens]
        x_attn = x_attn.reshape([x_size[0]*4, x_size[2]]) # Unpack each problem. [N problems, 4 sentences, maxseqlen tokens] -> [N*4 sentences, maxseqlen tokens]

        x = self.backbone(x_ipt, x_attn, return_dict=True) # Get representation for each image
        x = x.last_hidden_state[:, 0, :] # Get embedding of the [CLS] token for each sentence
        x = nn.functional.normalize(x, dim=1) # Normalize the resulting vectors

        x = x.reshape([-1, 4, self.backbone.config.hidden_size]) # Reshape into (# problems, 4 sentences per problem, BERT embed size

        x = (x[:,:,None,:] * x[:,None,:,:]).sum(3).sum(2) # None indexing unsqueezes another dimensions at that index
            # So you have (# problems, 4 sents per problem, 1, bert representation) * (# problems, 1, 4 sents per problem, bert representation)
            # Calculates the dot product of each mlp vector with every other vector (as well as itself) via broadcasting. Then sums the total
            # similarity
        x = -x # The odd one out's total similarity to every other vector should be LOWER than the other vector's, so negate the dot products, making argmax(x) = odd one out because it is the least negative
        return x

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument("--backbone", type=str, default='BERT_small')

        return parser
