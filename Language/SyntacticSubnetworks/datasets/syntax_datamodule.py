import os
import argparse
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
import torchvision
from torchvision import transforms as tvt

from datasets.base_datamodules import DataModuleBase

import pickle as pkl

class SyntaxDataModule(DataModuleBase):

    def __init__(
        self,
        data_dir,
        task,
        n_samples,
        tokenizer,
        max_seq_length,
        num_workers,
        batch_size,
        **kwargs,
    ):

        super(SyntaxDataModule, self).__init__( num_workers,
                                                batch_size,
        )

        print(task)

        self.train_set = SyntaxData(data_dir, task, tokenizer, split='train',max_seq_length=max_seq_length)
        self.val_set = SyntaxData(data_dir, task, tokenizer, split='val', max_seq_length=max_seq_length)
        self.test_set = SyntaxData(data_dir, task, tokenizer, split='test', max_seq_length=max_seq_length)

    @staticmethod
    def add_dataset_specific_args(parent_parser):

        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument('--data_dir', type=str, default='')
        parser.add_argument('--num_workers', type=int, default=0)
        parser.add_argument('--batch_size', type=int, default=None)
        parser.add_argument('--task', type=str, default='0')
        parser.add_argument('--n_samples', type=int, default=-1)
        parser.add_argument('--max_seq_length', type=int, default=50)

        return parser

TASKS={
    ### Subject Verb Agreement
    0: "sn_task_1_sv_singular",
    1: "sn_task_1_s_singular",
    2: "sn_task_1_v_singular",
    3: "sn_task_1_sv_singular_ablate_s",  # In this case, only the verb is plural in the OOO sentences
    4: "sn_task_1_sv_singular_ablate_v",  # In this case, only the subject is plural in the OOO sentences
    5: "sn_task_2_sv_plural",
    6: "sn_task_2_s_plural",
    7: "sn_task_2_v_plural",
    8: "sn_task_2_sv_plural_ablate_s",  # In this case, only the verb is singular in the OOO sentences
    9: "sn_task_2_sv_plural_ablate_v",  # In this case, only the subject is singular in the OOO sentences
    ### Reflexive Anaphora
    10: "sn_task_3_pro_ant_singular",
    11: "sn_task_3_pro_singular",
    12: "sn_task_3_ant_singular",
    13: "sn_task_3_pro_ant_singular_ablate_pro",  # In this case, only the antecedent is plural in the OOO sentences
    14: "sn_task_3_pro_ant_singular_ablate_ant",  # In this case, only the pronoun is plural in the OOO sentences
    15: "sn_task_4_pro_ant_plural",
    16: "sn_task_4_pro_plural",
    17: "sn_task_4_ant_plural",
    18: "sn_task_4_pro_ant_plural_ablate_pro",  # In this case, only the antecedent is singular in the OOO sentences
    19: "sn_task_4_pro_ant_plural_ablate_ant"  # In this case, only the pronoun is singular in the OOO sentences
}

Task2DataSizes = {
    0: {
        "train": 9500,
        "val": 500,
        "test": 1000
    },
    1: {
        "train": 9500,
        "val": 500,
        "test": 1000
    },
    2: {
        "train": 9500,
        "val": 500,
        "test": 1000
    },
    3: {
        "train": 0,
        "val": 300,
        "test": 300
    },    
    4: {
        "train": 0,
        "val": 300,
        "test": 300
    },
    5: {
        "train": 9500,
        "val": 500,
        "test": 1000
    },
    6: {
        "train": 9500,
        "val": 500,
        "test": 1000
    },
    7: {
        "train": 9500,
        "val": 500,
        "test": 1000
    },
    8: {
        "train": 0,
        "val": 300,
        "test": 300
    },    
    9: {
        "train": 0,
        "val": 300,
        "test": 300
    },
    10: {
        "train": 2500,
        "val": 200,
        "test": 200
    },
    11: {
        "train": 2500,
        "val": 200,
        "test": 200
    },
    12: {
        "train": 2500,
        "val": 200,
        "test": 200
    },
    13: {
        "train": 0,
        "val": 200,
        "test": 200
    },    
    14: {
        "train": 0,
        "val": 200,
        "test": 200
    },
    15: {
        "train": 2500,
        "val": 200,
        "test": 200
    },
    16: {
        "train": 2500,
        "val": 200,
        "test": 200
    },
    17: {
        "train": 2500,
        "val": 200,
        "test": 200
    },
    18: {
        "train": 0,
        "val": 200,
        "test": 200
    },    
    19: {
        "train": 0,
        "val": 200,
        "test": 200
    },
}

# Dataset

class SyntaxData(Dataset):

    def __init__(self, base_folder, task, tokenizer, split='train', max_seq_length=50):
        super().__init__()

        self.base_folder = base_folder
        self.task = TASKS[int(task)] # Only supports single task training

        self.split = split
        self.n_samples = Task2DataSizes[int(task)][split]

        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        if self.n_samples > 0:
            self._prepare_data()

    def _prepare_data(self):
        data_path = os.path.join(self.base_folder, self.task, self.split + ".pkl")
        data = pkl.load(open(data_path, "rb"))

        # Pretokenize all problems, append them to a list
        self.problem_encodings = []
        for problem in data:
            self.problem_encodings.append(self.tokenizer.batch_encode_plus(
                problem, max_length=self.max_seq_length, pad_to_max_length=True, trucation=True
                )
            )
        
    def __len__(self):
        return len(self.tasks) * self.n_samples

    def __getitem__(self, idx):
        # Get the BatchEncoding object corresponding to the idx,
        # The odd one out will always be the final sentence in these batches
        sample = self.problem_encodings[idx]
        return sample

