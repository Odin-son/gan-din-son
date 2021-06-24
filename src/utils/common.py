# functions to generate random data
import os.path as osp
import torch
import random

PRJ_DIR = osp.dirname(osp.dirname(osp.dirname(__file__)))
DATA_DIR = osp.join(PRJ_DIR, '../../dataset')


def generate_random_image(size):
    random_data = torch.rand(size)

    return random_data


def generate_random_seed(size):
    random_data = torch.randn(size)

    return random_data


def generate_random_one_hot(size):
    label_tensor = torch.zeros(size)
    random_idx = random.randint(0, size-1)
    label_tensor[random_idx] = 1.0

    return label_tensor
