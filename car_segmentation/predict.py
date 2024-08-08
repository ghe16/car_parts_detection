import torch
import torchvision.transforms as T
from PIL import Image
import torch.nn as nn

from utilidades.utilidades import accuracy, plot_mini_batch
from cargar_dataset import carga_carDataset
from models.UnetModel import *

import PIL
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_model(model_path):
    # Instantiate the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNET(3,4,2)
    model = torch.load(model_path)
    model = model.eval()
    
    return model

def preprocess_image(image_path):
    # Load the image and mask
    image = Image.open(image_path)
    
    # Convert to tensors
    transform_data = T.Compose([
    T.Resize([224,244]),
    T.ToTensor()
    ])
    image_tensor =  transform_data(image) # shape: [C, H, W]
    image_tensor= image_tensor.to(device=device, dtype = torch.float32)
    image_tensor = image_tensor.unsqueeze(0)
    return image_tensor


def apply_mask(model, image_tensor):
    # Expand mask to match the number of channels in the image
    #mask_tensor = mask_tensor.expand_as(image_tensor)
    print(image_tensor.shape)
    scores = model(image_tensor)
    preds = torch.argmax(scores, dim = 1).float()
    pred = preds.expand_as(image_tensor)
    img2 = image_tensor * pred
    
    # Apply the mask to the image
    #object_tensor = image_tensor * mask_tensor
    
    return img2

def save_result(tensor, output_path):
    # Convert tensor to PIL image
    print(tensor.shape)
    tensor = tensor.squeeze(0)
    result_image = T.ToPILImage()(tensor)
    
    # Save the result
    result_image.save(output_path)

def main():
    # Paths to the model, image, mask, and output
    model_path = "entire_model_.pth"
    image_path = 'imgs_prueba/car1.jpg'
    output_path = 'imgs_prueba/predicted_mask.jpg'
    
    # Load the model
    model = load_model(model_path)
    
    # Preprocess the input image and mask
    image_tensor = preprocess_image(image_path)
    
    # make the prediction
    object_tensor = apply_mask(model, image_tensor)
    
    # Save the result
    save_result(object_tensor, output_path)
    print(f"Result saved to {output_path}")

if __name__ == '__main__':
    main()
