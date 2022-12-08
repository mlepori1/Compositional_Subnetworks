
import argparse
import os
import copy
import pickle
import logging
import numpy as np

from PIL import Image


from data_generation.tasks import TASKS
from data_generation.generalization_tasks import TASKS as TASKS_GEN
from data_generation.utils import render_ooo


TASKS_IDX={
    ### elementary
    0: "task_shape",
    1: "task_pos",
    2: "task_size",
    3: "task_color",
    4: "task_rot",
    5: "task_flip",
    6: "task_count",
    7: "task_inside",
    8: "task_contact",
    ### compositions
    9: "task_sym_rot",
    10: "task_sym_mir",
    11: "task_pos_pos_1",
    12: "task_pos_pos_2",
    13: "task_pos_count_2",
    14: "task_pos_count_1",
    15: "task_pos_pos_4",
    16: "task_pos_count_3",
    17: "task_inside_count_1",
    18: "task_count_count",
    19: "task_shape_shape",
    20: "task_shape_contact_2",
    21: "task_contact_contact_1",
    22: "task_inside_inside_1",
    23: "task_inside_inside_2",
    24: "task_pos_inside_3",
    25: "task_pos_inside_1",
    26: "task_pos_inside_2",
    27: "task_pos_inside_4",
    28: "task_rot_rot_1",
    29: "task_flip_flip_1",
    30: "task_rot_rot_3",
    31: "task_pos_pos_3",
    32: "task_pos_count_4",
    33: "task_size_size_1",
    34: "task_size_size_2",
    35: "task_size_size_3",
    36: "task_size_size_4",
    37: "task_size_size_5",
    38: "task_size_sym_1",
    39: "task_size_sym_2",
    40: "task_color_color_1",
    41: "task_color_color_2",
    42: "task_sym_sym_1",
    43: "task_sym_sym_2",
    44: "task_shape_contact_3",
    45: "task_shape_contact_4",
    46: "task_contact_contact_2",
    47: "task_pos_size_1",
    48: "task_pos_size_2",
    49: "task_pos_shape_1",
    50: "task_pos_shape_2",
    51: "task_pos_rot_1",
    52: "task_pos_rot_2",
    53: "task_pos_col_1",
    54: "task_pos_col_2",
    55: "task_pos_contact",
    56: "task_size_shape_1",
    57: "task_size_shape_2",
    58: "task_size_rot",
    59: "task_size_inside_1",
    60: "task_size_contact",
    61: "task_size_count_1",
    62: "task_size_count_2",
    63: "task_shape_color",
    64: "task_shape_color_2",
    65: "task_shape_color_3",
    66: "task_shape_inside",
    67: "task_shape_inside_1",
    68: "task_shape_count_1",
    69: "task_shape_count_2",
    70: "task_rot_color",
    71: "task_rot_inside_1",
    72: "task_rot_inside_2",
    73: "task_rot_count_1",
    74: "task_color_inside_1",
    75: "task_color_inside_2",
    76: "task_color_contact",
    77: "task_color_count_1",
    78: "task_color_count_2",
    79: "task_inside_contact",
    80: "task_contact_count_1",
    81: "task_contact_count_2",
    82: "task_size_color_1",
    83: "task_size_color_2",
    84: "task_color_sym_1",
    85: "task_color_sym_2",
    86: "task_shape_rot_1",
    87: "task_shape_contact_5",
    88: "task_rot_contact_1",
    89: "task_rot_contact_2",
    90: "task_inside_sym_mir",
    91: "task_flip_count_1",
    92: "task_flip_inside_1",
    93: "task_flip_inside_2",
    94: "task_flip_color_1",
    95: "task_shape_flip_1",
    96: "task_rot_flip_1",
    97: "task_size_flip_1",
    98: "task_pos_rot_3",
    99: "task_pos_flip_1",
    100: "task_pos_flip_2",
    101: "task_flip_contact_1",
    102: "task_flip_contact_2",
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


def generate_dataset(task_name, task_fn, task_fn_gen, data_path='/media/data_cifs_lrs/projects/prj_visreason/cvrt_data/', image_size=128, seed=0, train_size=10000, val_size=500,  test_size=1000, test_gen_size=1000, fn_args="None"):
    # data_path = '/home/aimen/projects/cvrt_git/algs_images/'
    # data_path = '/media/data_cifs_lrs/projects/prj_visreason/cvrt_data/'

    task_path = os.path.join(data_path, task_name)

    n_train_samples_0 = 0
    n_train_samples_1 = train_size

    n_val_samples_0 = 0
    n_val_samples_1 = val_size

    n_test_samples_0 = 0
    n_test_samples_1 = test_size

    os.makedirs(task_path, exist_ok=True)
    os.makedirs(os.path.join(task_path,'train'), exist_ok=True)
    os.makedirs(os.path.join(task_path,'val'), exist_ok=True)
    os.makedirs(os.path.join(task_path,'test'), exist_ok=True)
    os.makedirs(os.path.join(task_path,'test_gen'), exist_ok=True)

    np.random.seed(seed)
    split = 'train'
    for i in range(n_train_samples_0, n_train_samples_1):
        if fn_args != "None":
            xy, size, shape, color = task_fn(fn_args)
        else:
            xy, size, shape, color = task_fn()

        # if not isinstance(shape, list):
        #     print('l')
        images = render_ooo(xy, size, shape, color, image_size=image_size)
        # images = render_ooo(*task_fn(), image_size=image_size)
        save_path = os.path.join(task_path, split, '{:05d}.png'.format(i))
        img = Image.fromarray(images).convert('RGB')
        img.save(save_path)

    np.random.seed(seed+1)
    split = 'val'
    for i in range(n_val_samples_0, n_val_samples_1):
        if fn_args != "None":
            images = render_ooo(*task_fn(fn_args), image_size=image_size)
        else:
            images = render_ooo(*task_fn(), image_size=image_size)
        save_path = os.path.join(task_path, split, '{:05d}.png'.format(i))
        img = Image.fromarray(images).convert('RGB')
        img.save(save_path)

    np.random.seed(seed+2)
    split = 'test'
    for i in range(n_test_samples_0, n_test_samples_1):
        if fn_args != "None":
            images = render_ooo(*task_fn(fn_args), image_size=image_size)
        else:
            images = render_ooo(*task_fn(), image_size=image_size)
        save_path = os.path.join(task_path, split, '{:05d}.png'.format(i))
        img = Image.fromarray(images).convert('RGB')
        img.save(save_path)

    np.random.seed(seed+2)
    if task_fn_gen != None:
        split = 'test_gen'
        for i in range(n_test_samples_0, n_test_samples_1):
            images = render_ooo(*task_fn_gen(), image_size=image_size)
            save_path = os.path.join(task_path, split, '{:05d}.png'.format(i))
            img = Image.fromarray(images).convert('RGB')
            img.save(save_path)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=0, help='seed for dataset generation')
    parser.add_argument('--data_dir', type=str, default='../cvrt_data/', help='directory to output dataset')
    parser.add_argument('--task_idx', default='a', help='index of the dataset')
    parser.add_argument('--train_size', type=int, default=10000, help='the number of training set samples')
    parser.add_argument('--val_size', type=int, default=500, help='the number of validation set samples')
    parser.add_argument('--test_size', type=int, default=1000, help='the number of test set samples')
    parser.add_argument('--test_gen_size', type=int, default=1000, help='the number of generalization test set samples')
    parser.add_argument('--image_size', type=int, default=128, help='image height and width in pixels')
    parser.add_argument('--task_args', type=str, default="None", help='Task specific arguments that control variation - only use for individual task datasets')

    ## for debugging
    ## python generate_dataset.py --data_dir ../CVR_dataset --task_idx a --train_size 4 --val_size 4 --test_size 4 --test_gen_size 4 --image_size 128

    args = parser.parse_args()

    pid = os.getpid()
    logging.info('JOB PID {}'.format(pid))

    task_idx = args.task_idx
    if task_idx == 'a':
        for i in range(0, 103):
            tn, tfn, _ = TASKS[i]
            _, tfn_g, _ = TASKS_GEN[i]
            generate_dataset(tn, tfn, tfn_g, args.data_dir, args.image_size, args.seed, args.train_size, args.val_size, args.test_size, args.test_gen_size, args.task_args)

    else:
        task_idx = int(task_idx)
        tn, tfn, _ = TASKS[task_idx]
        #_, tfn_g, _ = TASKS_GEN[task_idx]
        tfn_g = None
        generate_dataset(tn, tfn, tfn_g, args.data_dir, args.image_size, args.seed, args.train_size, args.val_size, args.test_size, args.test_gen_size, args.task_args)

