import numpy as np
import matplotlib.pyplot as plt
import os
import random
import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision import transforms as T
from torch.utils.data import DataLoader, random_split
#
import PIL
from PIL import Image
#
from cargar_dataset import carga_carDataset
from models.UnetModel import *


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

##Paths
PATH = "carvana-image-masking-challenge/"
TRAIN_PATH = "carvana-image-masking-challenge/train/"
TRAIN_MASKS_PATH="carvana-image-masking-challenge/train_masks/"
TEST_PATH = "carvana-image-masking-challenge/test/"
#
BATCH_SIZE = 32

transform_data = T.Compose([
    T.Resize([224,244]),
    T.ToTensor()
])


full_dataset = carga_carDataset.Car_Dataset(
    TRAIN_PATH,
    TRAIN_MASKS_PATH,
    img_transforms=transform_data,
    mask_transforms=transform_data
)

TRAIN_SIZE = int(len(full_dataset)*0.8)
VAL_SIZE = len(full_dataset) - TRAIN_SIZE

train_dataset, val_dataset = random_split(full_dataset, [TRAIN_SIZE,VAL_SIZE])

train_loader = DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True)
val_loader   = DataLoader(val_dataset,batch_size=BATCH_SIZE,shuffle=True)

imgs, masks = next(iter(train_loader))

# print(imgs.shape, masks.shape)


""" # test imprimir las 10 primeras
for i, (x,y) in enumerate(train_loader):
    print(i, x.shape, y.shape)
    if i == 9:
        break

#  visualizar los graficos de un batch
imgs, masks = next(iter(train_loader))
def plot_mini_batch(imgs, masks):
    plt.figure(figsize=(20,10))
    for i in range(BATCH_SIZE):
        plt.subplot(4,8, i+1)
        img = imgs[i,...].permute(1,2,0).numpy()
        mask = masks[i,...].permute(1,2,0).numpy()
        plt.imshow(img)
        plt.imshow(mask, alpha=0.5)
        plt.axis('Off')
        plt.tight_layout()
    plt.show()

plot_mini_batch(imgs,masks) """



def model_test():
    x = torch.randn((32,3,224,224))
    model = UNET(3, 64, 2)
    return model(x)

preds = model_test()

print(preds.shape)


def accuracy(model, loader):
    pass

