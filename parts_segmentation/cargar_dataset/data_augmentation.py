from torch.utils.data import Dataset
from torchvision import transforms as T
from PIL import Image,ImageOps
import os
import torch
import random
import json

class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, mask):
        if random.random() < self.p:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        return img, mask

class RandomRotation:
    def __init__(self, degrees):
        self.degrees = degrees

    def __call__(self, img, mask):
        angle = random.uniform(-self.degrees, self.degrees)
        img = img.rotate(angle)
        mask = mask.rotate(angle)
        return img, mask
    
class Resize:
    def __init__(self, size):
        self.size = size

    def __call__(self, img, mask):
        img = img.resize(self.size, Image.BILINEAR)
        mask = mask.resize(self.size, Image.NEAREST)
        return img, mask


class RandomCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, img, mask):
        i, j, h, w = T.RandomCrop.get_params(img, output_size=self.size)
        img = T.functional.crop(img, i, j, h, w)
        mask = T.functional.crop(mask, i, j, h, w)
        return img, mask
    
def joint_transform(img,mask):

    img,mask = RandomHorizontalFlip()(img,mask)
    img,mask = RandomRotation(20)(img,mask)
    #img,mask = RandomCrop((224,224))(img,mask)
    img,mask = Resize((256,256))(img,mask)
    return img, mask


def map_to_class_indices(mask, class_indices='data/train/categories.json'):
    clases = json.loads(open(class_indices,'r').read())
    #clases = clases['categories']
    clases_list = []
    colors_list = []

    for cls in clases:
        clases_list.append(cls["id"]-1)
        colors_list.append(cls["graycolor"])
    
    colors_list = torch.tensor(colors_list, dtype=torch.float32)
    mask_flat = mask.view(-1)
    mapped_mask = torch.zeros_like(mask_flat)

    for i, val in enumerate(mask_flat):
        differences = torch.abs(colors_list - val)
        nearest_class = clases_list[torch.argmin(differences)]
        mapped_mask[i] = nearest_class
    return mapped_mask.view(mask.shape)
