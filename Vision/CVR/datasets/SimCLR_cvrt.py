import os
import argparse
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
import torchvision
from torchvision import transforms
from datasets.base_datamodules import DataModuleBase

from PIL import Image

class ContrastiveTransformations(object):
    
    def __init__(self, base_transforms, n_views=2):
        self.base_transforms = base_transforms
        self.n_views = n_views
        
    def __call__(self, x):
        return [self.base_transforms(x) for i in range(self.n_views)]



class SimCLRDataModule(DataModuleBase):

    def __init__(
        self,
        data_dir,
        num_workers,
        batch_size,
        # image_size,
        **kwargs,
    ):

        super(SimCLRDataModule, self).__init__(   num_workers,
                                                batch_size,
        )

        transform = ContrastiveTransformations(self._default_transforms())

        self.train_set = SimClrCVRT(data_dir, split='train', image_size=128, transform=transform)
        self.val_set = SimClrCVRT(data_dir, split='val', image_size=128, transform=transform)

    def _default_transforms(self):
        # Set SimCLR Transformation
        #
        # Random Flip
        # Random Rotation
        # Random Affine Transformation
        # Random grayscale

        contrast_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter( contrast=0.5, 
                                        saturation=0.5)
            ], p=0.5),
            transforms.RandomApply([
                transforms.RandomAffine(
                    degrees=30,
                    translate=(.05, .05),
                    scale=(.9, 1),
                    shear=20,
                    fill=1
                )
            ], p=0.5),
            transforms.RandomRotation(degrees=90, fill=1),
            transforms.RandomGrayscale(p=0.2),
            transforms.Normalize((0.9, 0.9, 0.9), (0.1, 0.1, 0.1)), # Same as CVR
            ])
        return contrast_transforms

    @staticmethod
    def add_dataset_specific_args(parent_parser):

        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument('--data_dir', type=str, default='')
        parser.add_argument('--num_workers', type=int, default=0)
        parser.add_argument('--batch_size', type=int, default=None)

        return parser


TASKS={
    103: "sn_task_1_contact_inside",
    104: "sn_task_1_inside",
    105: "sn_task_1_contact",
    106: "sn_task_1_contact_inside_both_ooo",
    107: "sn_task_1_contact_inside_ablate_contact",
    108: "sn_task_1_contact_inside_ablate_inside",
    109: "sn_task_2_inside_count",
    110: "sn_task_2_inside",
    111: "sn_task_2_count",
    112: "sn_task_2_inside_count_both_ooo",
    113: "sn_task_2_inside_count_ablate_inside",
    114: "sn_task_2_inside_count_ablate_count",
    115: "sn_task_3_contact_count",
    116: "sn_task_3_contact",
    117: "sn_task_3_count",
    118: "sn_task_3_contact_count_both_ooo",
    119: "sn_task_3_contact_count_ablate_contact",
    120: "sn_task_3_contact_count_ablate_count",
}


# Dataset

class SimClrCVRT(Dataset):

    def __init__(self, base_folder, split='train', image_size=128, transform=None):
        super().__init__()

        self.base_folder = base_folder

        # All base tasks
        tasks = [103, 109, 115]
        self.tasks = [TASKS[t] for t in tasks]

        self.split = split

        if split == 'train':
            self.n_samples = 10000
        elif split == 'val':
            self.n_samples = 500

        self.image_size = image_size

        self.transform = transform
        self.totensor = transforms.ToTensor()

    def __len__(self):
        return len(self.tasks) * self.n_samples

    def __getitem__(self, idx):
        task_idx = idx // self.n_samples
        sample_idx = idx % self.n_samples

        sample_path = os.path.join(self.base_folder, self.tasks[task_idx], self.split, '{:05d}.png'.format(sample_idx))
        sample = Image.open(sample_path)

        sample = self.totensor(sample)
        # What is true height of the image, with padding
        im_size = sample.shape[1]
        # Padding is uniform around each data point
        pad = im_size - self.image_size

        # this line takes an image of 4 samples with some black padding, reshapes it such that
        # each image is a seperate element of the batch, permutes the dimensions, and then strips off the padding
        sample = sample.reshape([3, im_size, 4, im_size]).permute([2,0,1,3])[:, :, pad//2:-pad//2, pad//2:-pad//2]

        sample = sample[2:, :, : , :] # Take the last two images (the final one always is Odd One Out)
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, task_idx

