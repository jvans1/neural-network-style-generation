import torch
from collections import OrderedDict
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import pdb

mean   = [0.485, 0.456, 0.406]
stddev = [0.229, 0.224, 0.225]

def convert_to_average_pooling(modules):
    layers = []
    for i, layer in enumerate(modules):
        layer_name = "layer_"+str(i)
        if type(layer) is torch.nn.modules.pooling.MaxPool2d:
            layers.append((layer_name, nn.AvgPool2d((2, 2))))
        else:
            layers.append((layer_name, layer))
    return nn.Sequential(OrderedDict(layers))

class Truncate():
    def __init__(self, length):
        self.length = length

    def truncate(self, img):
        return img[:,:-self.length, :]

def get_image(index, truncate_length):
    truncater = Truncate(truncate_length)
    lam = lambda img: truncater.truncate(img)
    return ImageFolder("data/", transforms.Compose([
        transforms.Scale(224),
        transforms.ToTensor(),
        transforms.Lambda(lam),
        transforms.Normalize(mean=mean, std=stddev),
    ]))[index][0]

def unnormalize(img):
    #match dimensions (3,) -> (3,1,1)
    shaped_std = torch.Tensor(stddev).unsqueeze(1).unsqueeze(1)
    shaped_mean =  torch.Tensor(mean).unsqueeze(1).unsqueeze(1)
    return img * shaped_std + shaped_mean

def imshow(img):
    img = unnormalize(img)
    img = img.numpy()
    plt.imshow(np.transpose(img, (1,2,0)))


