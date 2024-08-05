import numpy as np
impfor matplotlib.pyplot as plt
import os
import random
import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision import transforms as T
from torch.utils.data import DataLoader, Dataset, random_split
#
import PIL
from PIL import Image
#
device = torch,device('cuda' if torch.cuda.is_avialable() else cpu)
print(device)

##Paths
PATH = "carvana-image-masking-challenge/"
TRAIN_PATH = "carvana-image-masking-challenge/train/"
TRAIN_MASKS_PATH="carvana-image-masking-challenge/train_masks/"
TEST_PATH = "carvana-image-masking-challenge/test/"

#creating the dataset
class Car_Dataset(Dataset):
    def __init__ (self,data,masks=None,img_transforms=None,mask_transforms=None):
        """
        data - train data patth
        masks - train masks path

        """

        self.train_data = data
        self.train_masks = masks
        
        self.img_transforms = img_transforms
        self.mask_transforms = mask_transforms

        self.images = sorted(os.listdir(self.train_data))
        self.masks = sorted(os.listdir(self.train_masks))

    def __len__(self):
        if self.train_masks is not None:
            assert len(self.images) == len(self.masks)  #not the same number of images and masks
        return len(self.images)
    
    def __getitem__(self,idx):
        image_name = os.path.join(self.train_data,self.images[idx]) #anadir ruta a imagen
        img = Image.open(image_name)
        if self.img_transforms is not None:
            img = self.img_transforms(img)     # aplicar transformaciones

        if self.train_masks is not None:
            mask_name = os.path.join(self.train_masks, self.masks[idx])    #cargamos la mascara


    def __getitem__(self, idx):