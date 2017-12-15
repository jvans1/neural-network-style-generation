import torch
from collections import OrderedDict
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import pdb

class Truncate():
    def __init__(self, length):
        self.length = length

    def truncate(self, img):
        return img[:,:-self.length, :]

    def left_trim(self, img):
        return img[:, :, self.length:]

    def right_trim(self, img):
        return img[:, :, :-self.length]



def load_raw(index):
    return ImageFolder("data/")[index][0]

def get_image(index, truncate_length, left_trim=None, side_length = None, center_crop=None, resize=None):
    tfs = [
        transforms.ToTensor()
        #  transforms.Normalize(mean=mean, std=stddev),
    ]

    if center_crop is not None:
        crop = transforms.CenterCrop(center_crop)
        tfs.insert(0, crop)

    if resize is not None:
        rsz = transforms.Scale(resize)
        tfs.insert(0, rsz)

    if truncate_length is not None:
        truncater = Truncate(truncate_length)
        lam = lambda img: truncater.truncate(img)
        tfs.append(lam)

    if left_trim is not None:
        truncater = Truncate(left_trim)
        lam = lambda img: truncater.left_trim(img)
        tfs.append(lam)

    if side_length is not None:
        truncater = Truncate(side_length)
        lam = lambda img: truncater.right_trim(img)
        tfs.append(lam)
    return ImageFolder("data/", transforms.Compose(tfs))[index][0]

def unnormalize(img):
    shaped_std = torch.Tensor(stddev).unsqueeze(1).unsqueeze(1)
    shaped_mean =  torch.Tensor(mean).unsqueeze(1).unsqueeze(1)
    return img * shaped_std + shaped_mean


unloader = transforms.ToPILImage()
def imshow(img):
    img = img.clone().cpu()
    img = unloader(img)
    plt.imshow(img)
