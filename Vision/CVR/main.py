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


from models.model import Model
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
        self.l0_masks = masks
        self.temp_increase = final_temp**(1./total_epochs)

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.l0_masks["backbone"]:
            temp = pl_module.backbone.get_temp()
            pl_module.backbone.set_temp(temp * self.temp_increase)
        elif self.l0_masks["mlp"]:
            temp = pl_module.mlp.get_temp()
            pl_module.mlp.set_temp(temp * self.temp_increase)


def cli_main():

    argv = sys.argv[1:]

    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--config", default=None, help="where to load YAML configuration", metavar="FILE")
    parser.add_argument('--exp_dir', type=str, default='../experiments/', help='experiment output directory')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument('--dataset', type=str, default='SVHNSupDataModule', help='dataset to use for training')
    parser.add_argument('--ckpt_period', type=int, default=3, help='save checkpoints every')
    parser.add_argument('--early_stopping', type=int, default=0, help='0 = no early stopping')
    parser.add_argument('--refresh_rate', type=int, default=10, help='progress bar refresh rate')
    parser.add_argument('--es_patience', type=int, default=40, help='early stopping patience')



    args = parse_args(parser, argv) # Here is where variables from the config file override command line args

    # trainer args
    parser = pl.Trainer.add_argparse_args(parser)

    # model args
    model_type = Model
    parser = model_type.add_model_specific_args(parser)

    # dataset args
    dataset_type = vars(datasets)[args.dataset]
    parser = dataset_type.add_dataset_specific_args(parser)

    args = parse_args(parser, argv)

    base_pretrained_weights = copy.deepcopy(args.pretrained_weights)
    base_train_masks = copy.deepcopy(args.train_masks)
    base_train_weights = copy.deepcopy(args.train_weights)

    model_id = 0
    df = pd.DataFrame()

    # Iterate through all training hyperparameters
    for seed in args.seed_list:
        for lr in args.lr_list:
            for batch_size in args.batch_size_list:
                for l0_stages in args.l0_stage_list:
                    for l0_init in args.l0_init_list:

                        # Increment model ID for next training
                        model_id += 1

                        # Reset pretrained weights, train weights and train mask from testing
                        args.pretrained_weights = copy.deepcopy(base_pretrained_weights)
                        args.train_masks = copy.deepcopy(base_train_masks)
                        args.train_weights = copy.deepcopy(base_train_weights)

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

                        os.makedirs(args.exp_dir, exist_ok=True)
                        os.makedirs(args.results_dir, exist_ok=True)
                        save_config(args.__dict__, os.path.join(args.exp_dir, 'config.yaml'))

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
                        

                        trainer.fit(model, datamodule)

                        # Load up best model if pretraining
                        best_model = model if model_checkpoint.best_model_path == "" else model_type.load_from_checkpoint(checkpoint_path=model_checkpoint.best_model_path)

                        trained_weights = {
                            "backbone": os.path.join(args.exp_dir, str(model_id) + '_backbone.pt'),
                            "mlp": os.path.join(args.exp_dir, str(model_id) + '_mlp.pt')
                        }


                        torch.save(best_model.backbone.state_dict(), trained_weights["backbone"])
                        torch.save(best_model.mlp.state_dict(), trained_weights["mlp"])
                        if best_model.task_embedding != None:
                            torch.save(best_model.task_embedding.state_dict(), os.path.join(args.exp_dir, str(model_id) + '_embedding.pt'))

                        metrics = metrics_callback.get_all()
                        if args.use_last == True:
                            # Get the last validation accuracy if we're using the last model
                            best_val_acc = metrics['metrics/val_acc'][-1]
                            best_val_loss = metrics['metrics/val_loss'][-1]

                        else:
                            best_val_loss = np.nanmax(metrics['metrics/val_loss'] + [0])
                            best_val_acc = metrics['metrics/val_acc'][np.nanargmax(metrics['metrics/val_loss']]

                        best_epoch = (np.nanargmax(metrics['metrics/val_loss'] + [0])+1) * args.ckpt_period

                        output_dict = {
                                '0_Model_ID': model_id,
                                '0_train': 1,
                                '0_train_task': args.train_task,
                                '0_ablation': '',
                                '0_exp_dir': args.exp_dir,
                                '0_model': args.model,
                                '0_seed': args.seed,
                                '0_dataset': args.dataset,
                                '0_LM_init': args.LM_init,
                                '0_freeze_until': args.freeze_until,
                                '1_task': args.task,
                                '2_val_acc': best_val_acc,
                                '2_val_loss': best_val_loss,
                                '2_best_epoch': best_epoch,
                                '2_used_last_model': args.use_last,
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
                        args.pretrained_weights = trained_weights

                        # When creating models, freeze model weights and mask weights
                        for key in args.train_masks.keys():
                            args.train_masks[key] = False

                        for key in args.train_weights.keys():
                            args.train_weights[key] = False

                        for ablation in args.ablation_strategies:
                            if args.seed is not None:
                                pl.seed_everything(args.seed)

                            if ablation == "none":
                                args.ablate_mask = None
                            else:
                                args.ablate_mask = ablation

                            test_model = model_type(**args.__dict__)

                            for task in args.test_tasks:

                                if args.seed is not None:
                                    pl.seed_everything(args.seed)

                                # Set the correct test task
                                args.task = task

                                # initializing the testing dataset
                                eval_datamodule = dataset_type(**args.__dict__)
                            
                                # During mask hparam search, only want to test on the validation sets
                                # This is to allow us to find the right masking parameters.
                                # Only during final testing do we want to see performance on test split
                                if args.evaluation_type == "test":
                                    trainer.test(model=test_model, datamodule=eval_datamodule)
                                    test_result = test_model.test_results
                                elif args.evaluation_type == "validate":
                                    # Maintain the same reporting logic during test and hparam search, just different data
                                    val_loader = eval_datamodule.val_dataloader()
                                    trainer.test(test_model, val_loader)
                                    test_result = test_model.test_results
                                
                                global_avg, per_task, per_task_avg = process_results(test_result, args.task)

                                output_dict = {
                                        '0_Model_ID': model_id,
                                        '0_train': 0,
                                        '0_train_task': args.train_task,
                                        '0_ablation': args.ablate_mask,
                                        '0_exp_dir': args.exp_dir,
                                        '0_model': args.model,
                                        '0_seed': args.seed,
                                        '0_dataset': args.dataset,
                                        '0_eval_type': args.evaluation_type,
                                        '1_task': args.task,
                                        '3_backbone': args.backbone,
                                        '3_batch_size': args.batch_size,
                                        '3_lr': args.lr,
                                        '3_l0_init': l0_init,
                                        '3_l0_stages': l0_stages
                                    }

                                output_dict.update({'2_'+k:v for k,v in global_avg.items()})
                                df = df.append(output_dict, ignore_index=True)

                                print("Saving csv")
                                # Will overwrite this file after every evaluation
                                df.to_csv(os.path.join(args.results_dir, 'results.csv'))

                        # Get rid of trained models after testing
                        if not args.save_models:
                            shutil.rmtree(args.exp_dir)

if __name__ == '__main__':
    print(os.getpid())

    cli_main()
