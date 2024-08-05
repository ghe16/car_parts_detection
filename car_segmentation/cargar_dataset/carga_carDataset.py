from torch.utils.data import Dataset
from torchvision import transforms as T
from PIL import Image
import os


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
        trans = T.ToTensor()
        if self.img_transforms is not None:
            img = self.img_transforms(img)     # aplicar transformaciones
        else:
            img = trans(img)

        if self.train_masks is not None:
            mask_name = os.path.join(self.train_masks, self.masks[idx])    #cargamos la mascara
            mask = Image.open(mask_name)
            if self.mask_transforms is not None:
                mask = self.mask_transforms(mask)
            else:
                mask = trans(mask)
            #normalizar las mascaras
            mask_max = mask.max().item()
            mask /= mask_max
        else:
            return img
        
        return img, mask
