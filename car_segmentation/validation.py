import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision import transforms as T

from utilidades.utilidades import accuracy, plot_mini_batch
from cargar_dataset import carga_carDataset
from models.UnetModel import *
from torch.utils.data import DataLoader, random_split

import PIL
from PIL import Image


def main():
    ##Paths
    PATH = "carvana-image-masking-challenge/"
    TRAIN_PATH = "carvana-image-masking-challenge/train/"
    TRAIN_MASKS_PATH="carvana-image-masking-challenge/train_masks/"
    TEST_PATH = "carvana-image-masking-challenge/test/"
    #
    BATCH_SIZE = 32
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load("entire_model_v1.pth")
    model = model.eval()
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
    _ , val_dataset = random_split(full_dataset, [TRAIN_SIZE,VAL_SIZE])
    val_loader   = DataLoader(val_dataset,batch_size=BATCH_SIZE,shuffle=True)

    imgs_val, masks_val = next(iter(val_loader))
    imgs_val, masks_val = imgs_val.to(device), masks_val.to(device) 
    model = model.to(device)
    with torch.no_grad():
        scores = model(imgs_val)
        preds = torch.argmax(scores, dim = 1).float()

    #imgs_val = imgs_val.cpu()
    #preds = preds.cpu()
    plt.figure(figsize=(20,10))
    print(scores.shape)
    print(imgs_val.shape) 
    print(preds.shape)
    for i in range(BATCH_SIZE):
        plt.subplot(4,8, i+1)
        img = imgs_val[i,...]
        pred = preds[i,...].expand_as(img)
        img2 = img * pred
        img2 = T.ToPILImage()(img2)
        plt.imshow(img2)
        #plt.imshow(pred,cmap=plt.cm.gray)
        plt.axis('Off')
        plt.tight_layout()
    plt.show()
    plt.savefig("validation.png")   




if __name__ == '__main__':
    main()
