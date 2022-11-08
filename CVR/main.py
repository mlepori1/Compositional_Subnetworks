import os
import sys
# import yaml sys
import argparse
import copy
import torch
import numpy as np
import pandas as pd

import pytorch_lightning as pl

from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks import TQDMProgressBar


import modules
import datasets

from utils import parse_args, save_config, find_best_epoch, process_results

class MetricsCallback(Callback):
    """PyTorch Lightning metric callback."""

    def __init__(self):
        super().__init__()
        self.metrics = []
        self.all_keys = []

    def on_validation_epoch_end(self, trainer, pl_module):

        each_me = {}
        for k,v in trainer.callback_metrics.items():
            each_me[k] = v.item()
            if k not in self.all_keys:
                self.all_keys.append(k)

        self.metrics.append(each_me)

    def get_all(self):

        all_metrics = {}
        for k in self.all_keys:
            all_metrics[k] = []

        for m in self.metrics[1:]:
            for k in self.all_keys:
                v = m[k] if k in m else np.nan
                all_metrics[k].append(v)

        return all_metrics

class TemperatureCallback(Callback):

    def __init__(self, total_epochs, final_temp, masks):
        # L0 MLP determines whether the MLP is getting trained or the backbone
        self.l0_masks = masks
        self.temp_increase = final_temp**(1./total_epochs)

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.l0_masks["backbone"]:
            temp = pl_module.backbone.get_temp()
            pl_module.backbone.set_temp(temp * self.temp_increase)
            print(pl_module.backbone.temp)
        elif self.l0_masks["mlp"]:
            temp = pl_module.mlp.get_temp()
            pl_module.mlp.set_temp(temp * self.temp_increase)
            print(pl_module.mlp.get_temp())


def cli_main():

    argv = sys.argv[1:]

    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--config", default=None, help="where to load YAML configuration", metavar="FILE")

    parser.add_argument('--exp_dir', type=str, default='../experiments/', help='experiment output directory')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument('--model', type=str, default='CNN_VAE', help='self supervised training method')
    parser.add_argument('--dataset', type=str, default='SVHNSupDataModule', help='dataset to use for training')
    parser.add_argument('--ckpt_period', type=int, default=3, help='save checkpoints every')
    parser.add_argument('--early_stopping', type=int, default=0, help='0 = no early stopping')
    parser.add_argument('--refresh_rate', type=int, default=10, help='progress bar refresh rate')
    parser.add_argument('--es_patience', type=int, default=40, help='early stopping patience')



    args = parse_args(parser, argv) # Here is where variables from the config file override command line args

    # trainer args
    parser = pl.Trainer.add_argparse_args(parser)

    # model args
    model_type = vars(modules)[args.model]
    parser = model_type.add_model_specific_args(parser)

    # dataset args
    dataset_type = vars(datasets)[args.dataset]
    parser = dataset_type.add_dataset_specific_args(parser)

    args = parse_args(parser, argv)

    base_pretrained_weights = args.pretrained_weights
    base_train_masks = args.train_masks
    base_train_weights = args.train_weights

    model_id = 0
    df = pd.DataFrame()

    # Iterate through all training hyperparameters
    for seed in args.seed_list:
        for lr in args.lr_list:
            for batch_size in args.batch_size_list:
                for l0_stages in args.l0_stage_list:
                    for l0_init in args.l0_init_list:

                        # Reset pretrained weights, train weights and train mask from testing
                        args.pretrained_weights = base_pretrained_weights
                        args.train_mask = base_train_masks
                        args.train_weights = base_train_weights

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

                        print(model.hparams)

                        fit_kwargs = {}

                        os.makedirs(args.exp_dir, exist_ok=True)
                        save_config(args.__dict__, os.path.join(args.exp_dir, 'config.yaml'))

                        # training

                        # Set up callbacks
                        logger = TensorBoardLogger(args.exp_dir, default_hp_metric=False)
                        model_checkpoint = pl.callbacks.ModelCheckpoint(dirpath=args.exp_dir, save_top_k=1, mode='max', monitor='metrics/val_acc', every_n_epochs=args.ckpt_period, save_last=True)
                        callbacks = [model_checkpoint]
                        if args.early_stopping!=0:
                            early_stopping = pl.callbacks.EarlyStopping(monitor='metrics/val_acc', mode='max', patience=args.es_patience, stopping_threshold=1.0, strict=False) #0.99
                            callbacks.append(early_stopping)
                        if args.train_masks["backbone"] or args.train_masks["mlp"]:
                            callbacks.append(TemperatureCallback(args.max_epochs, args.max_temp, args.train_masks))
                        callbacks.append(TQDMProgressBar(refresh_rate=args.refresh_rate))
                        metrics_callback = MetricsCallback()
                        callbacks.append(metrics_callback)

                        trainer = pl.Trainer.from_argparse_args(args, logger=logger, callbacks=callbacks)
                        
                        trainer.fit(model, datamodule, **fit_kwargs)

                        # Load up best model
                        best_model = model if model_checkpoint.best_model_path == "" else model_type.load_from_checkpoint(checkpoint_path=model_checkpoint.best_model_path)

                        pretrained_weights = {
                            "backbone": os.path.join(args.exp_dir, str(model_id) + '_backbone.pt'),
                            "mlp": os.path.join(args.exp_dir, str(model_id) + '_mlp.pt')
                        }

                        # Save it
                        torch.save(best_model.backbone.state_dict(), pretrained_weights["backbone"])
                        torch.save(best_model.mlp.state_dict(), pretrained_weights["mlp"])
                        if best_model.task_embedding != None:
                            torch.save(best_model.task_embedding.state_dict(), os.path.join(args.exp_dir, str(model_id) + '_embedding.pt'))

                        metrics = metrics_callback.get_all()
                        best_val_acc = np.nanmax(metrics['metrics/val_acc'] + [0])
                        best_epoch = (np.nanargmax(metrics['metrics/val_acc'] + [0])+1) * args.ckpt_period

                        output_dict = {
                                '0_Model_ID': model_id,
                                '0_train': 1,
                                '0_train_task': args.train_task,
                                '0_ablation': '',
                                '0_exp_dir': args.exp_dir,
                                '0_model': args.model,
                                '0_seed': args.seed,
                                '0_dataset': args.dataset,
                                '1_task': args.task,
                                '2_val_acc': best_val_acc,
                                '2_best_epoch': best_epoch,
                                '3_backbone': args.backbone,
                                '3_batch_size': args.batch_size,
                                '3_lr': args.lr,
                                '3_l0_init': l0_init,
                                '3_l0_stages': l0_stages
                            }

                        # Append results on val set
                        df = df.append(output_dict, ignore_index=True)

                        # Test in a variety of configurations

                        # Set pretrained_weights to create new models with different behavior
                        # using the weights we just trained
                        args.pretrained_weights = pretrained_weights

                        # When creating models, freeze model weights and mask weights
                        for key in args.train_masks.keys():
                            args.train_masks[key] = False

                        for key in args.train_weights.keys():
                            args.train_weights[key] = False

                        for task in args.test_tasks:
                            for ablation in args.ablation_strategies:

                                if args.seed is not None:
                                    pl.seed_everything(args.seed)

                                # Set the args
                                args.task = task

                                if ablation == "none":
                                    args.ablate_mask = None
                                else:
                                    args.ablate_mask = ablation

                                # initializing the dataset and model
                                test_datamodule = dataset_type(**args.__dict__)
                                test_model = model_type(**args.__dict__)

                                # Test using trainer from before
                                trainer.test(model=test_model, datamodule=test_datamodule)
                                train_result = model.test_results

                                global_avg, per_task, per_task_avg = process_results(train_result, args.task)

                                output_dict = {
                                        '0_Model_ID': model_id,
                                        '0_train': 0,
                                        '0_train_task': args.train_task,
                                        '0_ablation': args.ablate_mask,
                                        '0_exp_dir': args.exp_dir,
                                        '0_model': args.model,
                                        '0_seed': args.seed,
                                        '0_dataset': args.dataset,
                                        '1_task': args.task,
                                        '3_backbone': args.backbone,
                                        '3_batch_size': args.batch_size,
                                        '3_lr': args.lr,
                                        '3_l0_init': l0_init,
                                        '3_l0_stages': l0_stages
                                    }

                                output_dict.update({'2_'+k:v for k,v in global_avg.items()})
                                output_dict.update({'5_'+k:v for k,v in per_task_avg.items()})

                                df = df.append(output_dict, ignore_index=True)

                                print("Saving csv")
                                # Will overwrite this file after every evaluation
                                df.to_csv(os.path.join(args.exp_dir, 'results.csv'))

                        # Increment model ID for next training
                        model_id += 1

if __name__ == '__main__':
    print(os.getpid())

    cli_main()
