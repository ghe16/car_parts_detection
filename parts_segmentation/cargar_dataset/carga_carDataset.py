from torch.utils.data import Dataset
from torchvision import transforms as T
from PIL import Image
import os
import torch
import numpy as np
from cargar_dataset  import data_augmentation


#creating the dataset
class Car_Dataset(Dataset):
    def __init__ (self,data,masks=None, data_augmented: bool=False):
        """
        data - train data patth
        masks - train masks path

        """

        self.train_data = data
        self.train_masks = masks  # [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]
        self.data_augmented = data_augmented
        
        self.images = sorted(os.listdir(self.train_data))
        self.masks = sorted(os.listdir(self.train_masks))

    def __len__(self):
        if self.train_masks is not None:
            assert len(self.images) == len(self.masks)  #not the same number of images and masks
        return len(self.images)
    
    def __getitem__(self,idx):
        image_name = os.path.join(self.train_data,self.images[idx]) #anadir ruta a imagen
        img = Image.open(image_name).convert("RGB")
        mask_name = os.path.join(self.train_masks,self.masks[idx]) 
        mask = Image.open(mask_name)
        #print(mask.getextrema())
        


        if self.data_augmented == True:
            image, mask = data_augmentation.joint_transform(img, mask)
        else: 
            #transform_data = T.Compose([
            #T.Resize([256,256])
            #])
            #image = transform_data(img)
            #mask = transform_data(mask)
            image = img.resize((256,256),Image.BICUBIC)
            mask = mask.resize((256,256),Image.NEAREST)
        image = T.ToTensor()(image)
        mask = torch.tensor(np.array(mask), dtype=torch.long)
        #print(mask.max())
        #mask = data_augmentation.map_to_class_indices(mask=mask)

        
        return image, mask