import torch
import torchvision.transforms as T
from PIL import Image
import torch.nn as nn

from models.UnetModel import *

import PIL
from PIL import Image
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_model(model_path):
    # Instantiate the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #model = UNET(3,4,2)
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
    scores = model(image_tensor)
    preds = torch.argmax(scores, dim = 1).float()
    pred = preds.expand_as(image_tensor)
    img2 = image_tensor * pred
    return img2

def save_result(tensor, output_path):
    # Convert tensor to PIL image
    tensor = tensor.squeeze(0)
    result_image = T.ToPILImage()(tensor)
    # Save the result
    result_image.save(output_path)

def main(args):
    # Paths to the model, image, mask, and output
    # Load the model
    model = load_model(args.model_path)
    image_path = args.image_path
    output_path = args.output_path
    # Preprocess the input image and mask
    image_tensor = preprocess_image(image_path)
    # make the prediction
    object_tensor = apply_mask(model, image_tensor)
    
    # Save the result
    save_result(object_tensor, output_path)
    print(f"Result saved to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser( description="Programa para poner parametros de entrada")
    parser.add_argument('--image_path',type=str,required=True,help= "Path a la imagen que se quiere evaluar")
    parser.add_argument('--model_path',type=str,default='entire_model_v1.pth',help="path al modelo,(opcional)")
    parser.add_argument('--output_path',type=str,default='imgs_prueba/predicted_mask.jpg',help="nombre del archivo de salida (opcional)")
    args = parser.parse_args()
    main(args)
