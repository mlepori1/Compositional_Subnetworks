import os
import sys
import shutil
# import yaml sys
import argparse
import copy
import torch
import numpy as np
import pandas as pd
import copy

import pytorch_lightning as pl

from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks import TQDMProgressBar
from main import MetricsCallback, TemperatureCallback


import models.model import Model
import datasets

from utils import parse_args, save_config, find_best_epoch, process_results


def setup_args(config):
    parser = argparse.ArgumentParser()

    args = parse_args(parser, config=config) # Here is where variables from the config file override command line args

    # trainer args
    parser = pl.Trainer.add_argparse_args(parser)

    # model args
    model_type = Model
    parser = model_type.add_model_specific_args(parser)

    # dataset args
    dataset_type = vars(datasets)[args.dataset]
    parser = dataset_type.add_dataset_specific_args(parser)

    args = parse_args(parser, argv)
    return args, dataset_type, model_type

def setup_training(args, dataset_type, model_type):
    # return model, trainer, datamodule
    #
    # For testing, just need a single value for every element of these for loops
    # Leave the iteration for consistency with config format
    for seed in args.seed_list:
        for lr in args.lr_list:
            for batch_size in args.batch_size_list:
                for l0_stages in args.l0_stage_list:
                    for l0_init in args.l0_init_list:

                        # Increment model ID for next training
                        model_id += 1

                        args.task = args.train_task

                        args.lr = lr
                        args.batch_size = batch_size

                        args.seed = seed
                        if args.seed is not None:
                            pl.seed_everything(args.seed)

                        args.l0_stages = l0_stages
                        args.l0_init = l0_init

                        # initializing the dataset and model
                        datamodule = dataset_type(**args.__dict__)
                        model = model_type(**args.__dict__)

                        os.makedirs(args.exp_dir, exist_ok=True)
                        os.makedirs(args.results_dir, exist_ok=True)

                        # training

                        # Set up callbacks
                        logger = TensorBoardLogger(args.exp_dir, default_hp_metric=False)
                        
                        # Disable loading best model for experiments, as we want the final model for continuous sparsification
                        if args.use_last == True:
                            model_checkpoint = pl.callbacks.ModelCheckpoint(dirpath=args.exp_dir, save_top_k=1, monitor=None, save_last=True)
                            callbacks = [model_checkpoint]
                        else:
                            model_checkpoint = pl.callbacks.ModelCheckpoint(dirpath=args.exp_dir, save_top_k=1, mode='max', monitor='metrics/val_loss', every_n_epochs=args.ckpt_period, save_last=True)
                            callbacks = [model_checkpoint]
                        if args.early_stopping!=0:
                            early_stopping = pl.callbacks.EarlyStopping(monitor='metrics/val_loss', mode='max', patience=args.es_patience, stopping_threshold=1.0, strict=False) #0.99
                            callbacks.append(early_stopping)
                        if args.train_masks["backbone"] or args.train_masks["mlp"]:
                            callbacks.append(TemperatureCallback(args.max_epochs, args.max_temp, args.train_masks))
                        callbacks.append(TQDMProgressBar(refresh_rate=args.refresh_rate))
                        metrics_callback = MetricsCallback()
                        callbacks.append(metrics_callback)

                        trainer = pl.Trainer.from_argparse_args(args, logger=logger, callbacks=callbacks)
                        return model, trainer, datamodule

def setup_test_model(args, best_model, ablation):
    trained_weights = {
            "backbone": os.path.join(args.exp_dir, 'backbone.pt'),
            "mlp": os.path.join(args.exp_dir, 'mlp.pt')
        }

    torch.save(best_model.backbone.state_dict(), trained_weights["backbone"])
    torch.save(best_model.mlp.state_dict(), trained_weights["mlp"])
    if best_model.task_embedding != None:
        torch.save(best_model.task_embedding.state_dict(), os.path.join(args.exp_dir, 'embedding.pt'))

    # Set pretrained_weights to create new models with different behavior
    # using the weights we just trained
    args.pretrained_weights = trained_weights

    # When creating models, freeze model weights and mask weights
    for key in args.train_masks.keys():
        args.train_masks[key] = False

    for key in args.train_weights.keys():
        args.train_weights[key] = False

    if args.seed is not None:
        pl.seed_everything(args.seed)

    if ablation == "none":
        args.ablate_mask = None
    else:
        args.ablate_mask = ablation
    return args


def test_training():
    # Ensure that weights change while training, and that no mask_weights exist
    config = "configs/tests/test_weight_train.yaml"


    for backbone_model in ["resnet50", "wideresnet50", "ViT"]:

        args, dataset_type, model_type = setup_args(config)
        args.backbone = backbone_model

        model, trainer, datamodule = setup_training(args, dataset_type, model_type)
    
        ### TEST: No masks initialized in the networks
        for layer in model.backbone.modules():
            assert hasattr(layer, "mask_weight") == False

        for layer in model.mlp.modules():
            assert hasattr(layer, "mask_weight") == False

        ## TEST: Weights change after training
        mlp_weight = copy.deepcopy(model.mlp.model[0].weight.data)
        if args.backbone == "resnet50" or args.backbone == "wideresnet50":
            bb_weight = copy.deepcopy(model.backbone.conv1.weight.data)
        elif args.backbone == "ViT":
            bb_weight = copy.deepcopy(model.backbone.encoder.layer[0].intermediate.dense.weight.data)

        trainer.fit(model, datamodule)
        best_model = model if model_checkpoint.best_model_path == "" else model_type.load_from_checkpoint(checkpoint_path=model_checkpoint.best_model_path)

        assert mlp_weight != best_model.mlp.model[0].weight.data
        if args.backbone == "resnet50" or args.backbone == "wideresnet50":
            assert bb_weight != best_model.backbone.conv1.weight.data
        elif args.backbone == "ViT":
            assert bb_weight != best_model.backbone.encoder.layer[0].intermediate.dense.weight.data

        ablation = "none"

        args = setup_test_model(args, best_model, ablation)

        test_model = model_type(**args.__dict__)

        ## Test test_model intialization
        assert test_model.mlp.model[0].weight.data == best_model.mlp.model[0].weight.data
        if args.backbone == "resnet50" or args.backbone == "wideresnet50":
            assert test_model.backbone.conv1.weight.data == best_model.backbone.conv1.weight.data
        elif args.backbone == "ViT":
            assert test_model.backbone.encoder.layer[0].intermediate.dense.weight.data == best_model.backbone.encoder.layer[0].intermediate.dense.weight.data

def test_resnet_mask_training():
    # Ensure that mask weights change during training, and that regular weights don't
    config = "configs/tests/RN_test_mask_train.yaml"

    args = parse_args(parser, config=config) # Here is where variables from the config file override command line args
    
    for backbone_model in ["resnet50", "wideresnet50"]:
        args, dataset_type, model_type = setup_args(config)

        args.backbone = backbone_model

        model, trainer, datamodule = setup_training(args, dataset_type, model_type)
                            
        ## TEST: Weights stay the same after training, Mask weights change
        mlp_weight = copy.deepcopy(model.mlp.model[0].weight.data)
        mlp_mask = copy.deepcopy(model.mlp.model[0].mask_weight.data)

        if args.backbone == "resnet50" or args.backbone == "wideresnet50":
            bb_weight = copy.deepcopy(model.backbone.conv1.weight.data)
            bb_mask = copy.deepcopy(model.backbone.conv1.mask_weight.data)

        trainer.fit(model, datamodule)
        best_model = model if model_checkpoint.best_model_path == "" else model_type.load_from_checkpoint(checkpoint_path=model_checkpoint.best_model_path)

        assert mlp_weight == best_model.mlp.model[0].weight.data
        if args.backbone == "resnet50" or args.backbone == "wideresnet50":
            assert bb_weight == best_model.backbone.conv1.weight.data
        assert mlp_mask != best_model.mlp.model[0].mask_weight.data
        if args.backbone == "resnet50" or args.backbone == "wideresnet50":
            assert bb_mask != best_model.backbone.conv1.mask_weight.data

def test_vit_mask_training():
        # Ensure that weights change while training, and that no mask_weights exist
    config = "configs/tests/ViT_test_mask_train.yaml"
 
    args, dataset_type, model_type = setup_args(config)

    model, trainer, datamodule = setup_training(args, dataset_type, model_type)
                        
    ## TEST: Weights stay the same after training, Mask weights change
    mlp_weight = copy.deepcopy(model.mlp.model[0].weight.data)
    mlp_mask = copy.deepcopy(model.mlp.model[0].mask_weight.data)

    bb_weight = copy.deepcopy(model.backbone.encoder.layer[0].intermediate.dense.weight.data)
    bb_mask = copy.deepcopy(model.backbone.encoder.layer[0].intermediate.dense.mask_weight.data)

    trainer.fit(model, datamodule)
    best_model = model if model_checkpoint.best_model_path == "" else model_type.load_from_checkpoint(checkpoint_path=model_checkpoint.best_model_path)

    assert mlp_weight == best_model.mlp.model[0].weight.data
    assert mlp_mask != best_model.mlp.model[0].mask_weight.data

    assert bb_weight == best_model.backbone.encoder.layer[0].intermediate.dense.weight.data
    assert bb_mask != best_model.backbone.encoder.layer[0].intermediate.dense.mask_weight.data

def test_resnet_mask_configuration():
    # Test L0 configurations
    # Stage 3 neither masks nor weights should change
    # Stage 4 masks should change, weights should not
    
    config = "configs/tests/RN_test_l0_configs.yaml"
    
    for backbone_model in ["resnet50", "wideresnet50"]:
        args, dataset_type, model_type = setup_args(config)

        args.backbone = backbone_model

        model, trainer, datamodule = setup_training(args, dataset_type, model_type)

        ## TEST: Stage 3 is static, stage 4 masks change
        for layer in model.backbone.layer3.modules():
            assert hasattr(layer, "mask_weight") == False
        
        bb_weight_3 = copy.deepcopy(model.backbone.layer3[0].conv1.weight.data)
        bb_mask_4 = copy.deepcopy(model.backbone.layer4[0].conv1.mask_weight.data)
        bb_weight_4 = copy.deepcopy(model.backbone.layer4[0].conv1.weight.data)

        trainer.fit(model, datamodule)
        best_model = model if model_checkpoint.best_model_path == "" else model_type.load_from_checkpoint(checkpoint_path=model_checkpoint.best_model_path)

        assert bb_weight_3 == best_model.backbone.layer3[0].conv1.weight.data
        assert bb_mask_4 != best_model.backbone.layer4[0].conv1.mask_weight.data
        assert bb_weight_4 == best_model.backbone.layer4[0].conv1.weight.data

def test_vit_mask_configuration():
    # Test L0 configurations
    # Layer 3 neither masks nor weights should change
    # Layer 4 masks should change, weights should not
    
    config = "configs/tests/ViT_test_l0_configs.yaml"

    args, dataset_type, model_type = setup_args(config)

    model, trainer, datamodule = setup_training(args, dataset_type, model_type)

    ## TEST: Layer 3 is static, Layer 4 masks change
    for layer in model.backbone.encoder.layer[3].modules():
        assert hasattr(layer, "mask_weight") == False
    
    bb_weight_3 = copy.deepcopy(model.backbone.encoder.layer[2].attention.attention.query.weight.data)
    bb_weight_4 = copy.deepcopy(model.backbone.encoder.layer[3].attention.attention.query.weight.data)
    bb_mask_4 = copy.deepcopy(model.backbone.encoder.layer[3].attention.attention.query.mask_weight.data)

    trainer.fit(model, datamodule)
    best_model = model if model_checkpoint.best_model_path == "" else model_type.load_from_checkpoint(checkpoint_path=model_checkpoint.best_model_path)
    
    assert bb_weight_3 == best_model.backbone.encoder.layer[2].attention.attention.query.weight.data
    assert bb_weight_4 == best_model.backbone.encoder.layer[3].attention.attention.query.weight.data
    assert bb_mask_4 != best_model.backbone.encoder.layer[3].attention.attention.query.mask_weight.data

def test_subnetwork_test_config_mask():
    # Test that hard mask results in binary mask
    config = "configs/tests/test_weight_train.yaml"

    for backbone_model in ["resnet50", "wideresnet50", "L0vit"]:
        args, dataset_type, model_type = setup_args(config)

        args.backbone = backbone_model

        model_id = 0
        
        model, trainer, datamodule = setup_training(args, dataset_type, model_type)

        trainer.fit(model, datamodule)
        best_model = model if model_checkpoint.best_model_path == "" else model_type.load_from_checkpoint(checkpoint_path=model_checkpoint.best_model_path)

        ablation = "none"

        args = setup_test_model(args, best_model, ablation)

        test_model = model_type(**args.__dict__)

        ## TEST: Assert that testing mask is binary
        # assert that testing model actually has mask_weights
        # assert that the model weights were actually loaded, not reinitialized, 
        # by comparing the values to the L0 init value
        mask_count = 0
        for layer in test_model.backbone.modules():
            if hasattr(layer, "mask_weight") == True:
                mask_count += 1
                mask = layer.compute_mask()
                assert torch.sum(mask == 1) + torch.sum(mask == 0) == len(mask.reshape(-1))
                assert torch.sum(layer.mask_weight.data == layer.mask_init_value) != len(layer.mask_weight.data.reshape(-1))
        assert mask_count != 0

        mask_count = 0
        for layer in test_model.mlp.modules():
            if hasattr(layer, "mask_weight") == True:
                mask_count += 1
                mask = layer.compute_mask()
                assert torch.sum(mask == 1) + torch.sum(mask == 0), len(mask.reshape(-1))
                assert torch.sum(layer.mask_weight.data == layer.mask_init_value) != len(layer.mask_weight.data.reshape(-1))
        assert mask_count != 0


def test_zero_ablation_mask():
    # Test that zero ablation results in inverse hard mask
    config = "configs/tests/test_weight_train.yaml"

    for backbone_model in ["resnet50", "wideresnet50", "L0vit"]:
        args, dataset_type, model_type = setup_args(config)

        args.backbone = backbone_model

        model_id = 0

        model, trainer, datamodule = setup_training(args, dataset_type, model_type)

        trainer.fit(model, datamodule)
        best_model = model if model_checkpoint.best_model_path == "" else model_type.load_from_checkpoint(checkpoint_path=model_checkpoint.best_model_path)

        ablation = "zero"

        args = setup_test_model(args, best_model, ablation)

        ablate_model = model_type(**args.__dict__)

        args.ablate_mask = None
        none_model = model_type(**args.__dict__)


        ## TEST: Assert that zero ablation is inverse of no ablation
        if args.backbone == "resnet50" or args.backbone == "wideresnet50":
            none_mask = none_model.bacbone.layer3[0].conv1.compute_mask()
            ablate_mask = ablate_model.backbone.layer3[0].conv1.compute_mask()
            assert (none_mask == 1) == (ablate_mask == 0)
            assert (none_mask == 0) == (ablate_mask == 1)

        if args.backbone == "L0vit":
            none_mask = none_model.backbone.encoder.layer[2].attention.attention.query.compute_mask()
            ablate_mask = ablate_model.backbone.encoder.layer[2].attention.attention.query.compute_mask()
            assert (none_mask == 1) == (ablate_mask == 0)
            assert (none_mask == 0) == (ablate_mask == 1)

def test_random_ablation_mask():
    # Test that random mask results in inverse hard mask, then random values
    config = "configs/tests/test_weight_train.yaml"

    for backbone_model in ["resnet50", "wideresnet50", "L0vit"]:
        args, dataset_type, model_type = setup_args(config)

        args.backbone = backbone_model

        model_id = 0

        model, trainer, datamodule = setup_training(args, dataset_type, model_type)

        trainer.fit(model, datamodule)
        best_model = model if model_checkpoint.best_model_path == "" else model_type.load_from_checkpoint(checkpoint_path=model_checkpoint.best_model_path)

        ablation = "random"

        args = setup_test_model(args, best_model, ablation)

        ablate_model = model_type(**args.__dict__)

        args.ablate_mask = None
        none_model = model_type(**args.__dict__)


        ## TEST: assert that all masked layers have random weight too
        for layer in ablate_mask.modules():
            if hasattr(layer, "mask_weight"):
                assert hasattr(layer, "random_weight")

        ## TEST: Assert that random ablation mask is inverse of no ablation
        if args.backbone == "resnet50" or args.backbone == "wideresnet50":
            none_mask = none_model.bacbone.layer3[0].conv1.compute_mask()
            ablate_mask = ablate_model.backbone.layer3[0].conv1.compute_mask()
            assert (none_mask == 1) == (ablate_mask == 0)
            assert (none_mask == 0) == (ablate_mask == 1)
            
            ablate_layer = ablate_model.backbone.layer3[0].conv1
            masked_weight = ablate_layer.weight * ablate_layer.mask # This will give you the inverse weights, 0's for ablated weights

            ## TEST: Assert that adding in random values is done correctly
            assert ablate_mask == masked_weight
            before_random = copy.deepcopy(masked_weight)
            masked_weight += (~ablate_layer.mask.bool()).float() * ablate_layer.random_weight # Invert the mask to target the remaining weights, make them random
            assert masked_weight * ablate_mask == before_random # Don't change any values except those in the subnetwork
            assert len(torch.nonzero(masked_weight)) == 0 # No 0 values now


        if args.backbone == "L0vit":
            none_mask = none_model.backbone.encoder.layer[2].attention.attention.query.compute_mask()
            ablate_mask = ablate_model.backbone.encoder.layer[2].attention.attention.query.compute_mask()
            assert (none_mask == 1) == (ablate_mask == 0)
            assert (none_mask == 0) == (ablate_mask == 1)

            ablate_layer = ablate_model.backbone.encoder.layer[2].attention.attention.query
            masked_weight = ablate_layer.weight * ablate_layer.mask # This will give you the inverse weights, 0's for ablated weights

            ## TEST: Assert that adding in random values is done correctly
            assert ablate_mask == masked_weight
            before_random = copy.deepcopy(masked_weight)
            masked_weight += (~ablate_layer.mask.bool()).float() * ablate_layer.random_weight # Invert the mask to target the remaining weights, make them random
            assert masked_weight * ablate_mask == before_random # Don't change any values except those in the subnetwork
            assert len(torch.nonzero(masked_weight)) == 0 # No exactly 0 values now