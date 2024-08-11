import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision import transforms as T

from utilidades.utilidades import accuracy
from cargar_dataset import carga_carDataset
from models.UnetModelMultiClass import *
from torch.utils.data import DataLoader, random_split

import PIL
from PIL import Image


def main():
    ##Paths
    PATH = "data/"
    TRAIN_PATH = "data/train/JPEGImages/"
    TRAIN_MASKS_PATH="data/train/JPEGMasks/"
    TEST_PATH = "data/test/"
    #
    BATCH_SIZE = 32
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNET(3,4,21)
    model = torch.load("entire_model_multi.pth")
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

    imgs_val = imgs_val.cpu().float()
    preds = preds.cpu()
    plt.figure(figsize=(20,10))
    print(scores.shape)
    print(imgs_val.shape) 
    print(preds.shape)
    for i in range(BATCH_SIZE):
        plt.subplot(4,8, i+1)
        img = imgs_val[i,...].permute(1,2,0).numpy()
        plt.imshow(img)
        plt.axis('Off')
        for j in scores[i,...]:
            mask_ = scores[i,j,...].permute(1,2,0).numpy()
            #img2 = img2
            #img2 = T.ToPILImage()(img2)
            plt.imshow(mask_)
        #plt.imshow(pred,cmap=plt.cm.gray)
        plt.axis('Off')
        plt.tight_layout()
        break
    plt.show()
    plt.savefig("validation.png")   




if __name__ == '__main__':
    main()
