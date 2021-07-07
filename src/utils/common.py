# functions to generate random data
import os
import os.path as osp
import torch
import torch.nn as nn
import random

PRJ_DIR = osp.dirname(osp.dirname(osp.dirname(__file__)))
DATA_DIR = osp.join(PRJ_DIR, 'dataset')
MODEL_DIR = osp.join(PRJ_DIR, 'models')


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


# modified from https://github.com/pytorch/vision/issues/720
class View(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape,

    def forward(self, x):
        return x.view(*self.shape)


# crop (numpy array) image to given width and height
def crop_centre(img, new_width, new_height):
    height, width, _ = img.shape
    startx = width//2 - new_width//2
    starty = height//2 - new_height//2
    return img[starty:starty + new_height, startx:startx + new_width, :]
