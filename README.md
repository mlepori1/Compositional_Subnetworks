# Compositional Subnetworks

## Overview
This repository contains the code needed to reproduce Break It Down: Evidence for Structural Compositionality in Neural Networks. There are three important top-level directories: `Language`, where all language experiments are implemented, `Vision`, where all vision experiments are implemented, and `transformers`, which is a copy of the Huggingface Transformers repository, with models edited to support continuous sparsification.

## Language
This directory contains a copy of Marvin & Linzen repository, `LM_Syneval`, which is used to create the language datasets used in this paper. Our results files are found under `Results`. 

The code used to run experiments is found under `Syntactic Subnetworks`. The most important files here are `model.py` and `main.py`. `model.py` implements a pytorch lightning wrapper that predicts odd-one-out stimuli using pairwise cosine similarities of embeddings generated from neural models. It also implements pruning functionality and creates models from config files.
`main.py` can be used to train a model, train masks over a model, and also to evaluate the subnetworks that a trained mask uncovers. This functionality is defined by a config file, which is an argument to `main.py`. Config files are found in the `configs` subdirectory. There is one subdirectory for each experiment, which contains subdirectories for each model that we evaluated (either pretrained or from-scratch BERT-small). These files can be used to reproduce our training runs and hyperparameter sweeps.

To run a test specified by a config file `$CONFIG`, simply run `python main.py --config $CONFIG`.
## Vision
This directory contains a heavily-edited version of the repository from "A Benchmark for Compositional Visual Reasoning" by Zerroug et al. (under `CVR`), and directory containing all of our results files (under `Results`). The `CVR` directory contains the code used to generate our vision datasets, as described in the README. We have edited the original repositories data generation scripts to add our new odd-one-out tests.

This directory is organized similarly to `Language`. `models` is a subdirectory defining the relevant models for our vision experiments. In particular, it implements a version of ResNet50 with continuous sparsification functionality. It also implements a wrapper that performs odd-one-out classification. Finally, it defines a pytorch lightning module that performs constrastive learning using the SimCLR algorithm. This model enables our pretrained vision experiments.
`SimCLR.py` is a simple script that pretrains a SimCLR model. `main.py` can be used to train a model on the odd-one-out task, train mask parameters, and evaluate the subnetworks that a trained mask uncovers. Config files are found in the `configs` subdirectory. There is one subdirectory for each experiment, which contains subdirectories for each model that we evaluated (ResNet50, WideResNet50, ResNet50 + SimCLR, ViT). These files can be used to reproduce our training runs and hyperparameter sweeps.

## Config Directory Structures
For a given task (i.e. **inside-contact**), the config directory is structured as follows:
- `Model_Training`: This contains one config file per model, which trains 3 models with the best hyperparameters for that architecture according to its hyperparameter sweep
- `Model1`: A subdirectory corresponding to a particular model.
  - `Model_Hyperparameters`: A subdirectory containing a config file that runs a hyperparameter sweep over learning rate and batch size for the odd-one-out task
  - `Mask_Hyperparameters`: A subdirectory containing config files for hyperparameters for each trained model. These files search over different mask configurations, mask intializations, and learning rates, for both subroutines (i.e. **inside** and **contact**)
  - `Mask_Training`: A subdirectory that contains the configurations relevant for actually generating the experimental results
    - `Mask_Configs`: For each model-task combination, these configurations generate three runs of mask training, using the best hyperparameters identified during the mask hyperparameter sweep
    - `Random_Configs`: Contains configurationst the run mask training over a randomly initialized model, using the same hyperparameters as the corresponding `Mask_Configs` file.
    - `Sparsity_Configs`: Configurations used when generating the layer-by-layer sparsity data, which is reported in the Appendix.
    
Inside the Resnet50 directories, the following subdirectories are included to reproduce our extended analysis assessing structural compositionality in neural networks that have already been pruned. These results are reported in the Appendix.
  - `Pruning_Hyperparameters`: A subdirectory containing config files that search over pruning parameters for each trained base model
  - `Pruning_Training`: After finding the best pruning parameters, trains binary masks and saves one pruned version of each base model
  - `Pruned_Mask_Hyperparameters`: Analogous to `Mask_Hyperparameters`, but for pruned models.
  - `Pruned_Mask_Training`: Analogous to `Mask_Training`, but for pruned models.
## Transformers
This is a copy of the Huggingface Transformers repository, where we edit ViT and BERT models to enable pruning. The relevant edited files are `transformers/src/transformers/models/modeling_bert.py` and `transformers/src/transformers/models/modeling_vit.py`. Make sure to install this local version of `transformers` before running the scripts found in the other directories!


