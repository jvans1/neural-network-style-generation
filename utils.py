import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import pdb

mean   = [0.485, 0.456, 0.406]
stddev = [0.229, 0.224, 0.225]
def top(img):
    return img[:,:-91, :]

folder = ImageFolder("data/", transforms.Compose([
    transforms.Scale(224),
    transforms.ToTensor(),
    transforms.Lambda(top),
    transforms.Normalize(mean=mean, std=stddev),
]))

def unnormalize(img):
    #match dimensions (3,) -> (3,1,1)
    shaped_std = torch.Tensor(stddev).unsqueeze(1).unsqueeze(1)
    shaped_mean =  torch.Tensor(mean).unsqueeze(1).unsqueeze(1)
    return img * shaped_std + shaped_mean

def imshow(img):
    img = unnormalize(img)
    img = img.numpy()
    plt.imshow(np.transpose(img, (1,2,0)))


