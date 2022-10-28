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


import modules
import datasets

from utils import parse_args, save_config, process_results


def cli_main():

    argv = sys.argv[1:]

    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--config", default=None, help="where to load YAML configuration", metavar="FILE")

    args = parse_args(parser, argv) # Here is where variables from the config file override command line args

    ablation_strategies = args.ablation_strategies
    tasks = args.tasks
    
    df = pd.DataFrame()

    # Iterate through different test settings for your base model configuration

    # trainer args
    parser = pl.Trainer.add_argparse_args(parser)

    # model args
    model_type = vars(modules)[args.model]
    parser = model_type.add_model_specific_args(parser)

    # dataset args
    dataset_type = vars(datasets)[args.dataset]
    parser = dataset_type.add_dataset_specific_args(parser)

    args = parse_args(parser, argv)

    for task in tasks:
        for ablation in ablation_strategies:
            if args.seed is not None:
                pl.seed_everything(args.seed)

            # Set the args
            args.task = task

            if ablation == "none":
                args.ablate_mask = None
            else:
                args.ablate_mask = ablation

            # initializing the dataset and model
            datamodule = dataset_type(**args.__dict__)
            model = model_type(**args.__dict__)

            print(model.hparams)

            os.makedirs(args.exp_dir, exist_ok=True)
            os.makedirs(args.path_db, exist_ok=True)
            save_config(args.__dict__, os.path.join(args.exp_dir, 'config.yaml'))

            # Set up trainer
            logger = TensorBoardLogger(args.exp_dir, default_hp_metric=False)

            trainer = pl.Trainer.from_argparse_args(args, logger=logger)

            trainer.test(model=model, datamodule=datamodule)
            train_result = model.test_results

            global_avg, per_task, per_task_avg = process_results(train_result, args.task)

            output_dict = {
                '0_train': 0,
                '0_ablation': args.ablate_mask,
                '0_exp_name': args.exp_name,
                '0_exp_dir': args.exp_dir,
                '0_model': args.model,
                '0_seed': args.seed,
                '0_dataset': args.dataset,
                '1_task': args.task,
                '3_backbone': args.backbone,
                '3_batch_size': args.batch_size,
            }

            output_dict.update({'2_'+k:v for k,v in global_avg.items()})
            output_dict.update({'5_'+k:v for k,v in per_task_avg.items()})

            df = df.append(output_dict, ignore_index=True)

    print("Saving csv")
    df.to_csv(os.path.join(args.path_db, args.exp_name + '_db.csv'))

if __name__ == '__main__':
    print(os.getpid())

    cli_main()
