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
from utilidades.utilidades import finding_lr,accuracy, model_test


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



# print(imgs.shape, masks.shape)


""" # test imprimir las 10 primeras
for i, (x,y) in enumerate(train_loader):
    print(i, x.shape, y.shape)
    if i == 9:
        break
"""




#training

def train(model, optimiser, scheduler = None, epochs = 100, store_every = 25):
    model = model.to(device= device)
    for epoch in range(epochs):
        train_correct_num = 0   # se resetea cada epoch
        train_total =  0   # num elementos evaluados
        train_cost_acum = 0. #  coste acumulado
        for mb , (x, y) in enumerate(train_loader, start=1):
            model.train()
            x=x.to(device=device, dtype = torch.float32)
            y = y.to(device=device, dtype= torch.float32).squeeze(1)  # eliminamos el canal 1 (blanco y negro)
            #print("x",x.shape, "y", y.shape)
            _, targety, targetx = y.shape
            scores  = model(x)
            scores = torch.argmax(scores,dim=1).float()
            batches,  height, width = scores.shape
            start_Y = (height - targety) // 2
            start_X = (width - targetx) // 2
            scores =  scores[:,start_Y:start_Y + targety, start_X:start_X + targetx]
            #print("scores_new",scores.shape)
            cost = F.cross_entropy(input=scores, target=y)
            #cost = criterion(scores, y)
            optimiser.zero_grad()
            cost.requires_grad = True
            cost.backward()
            optimiser.step()
            if scheduler: 
                scheduler.step()
            #train_predictions = torch.argmax(scores, dim = 1)
            #train_predictions = train_predictions[start_Y-1:start_Y + targety, start_X-1:start_X + targetx]
            #print(train_predictions.shape, "y", y.shape)
            train_correct_num += (scores == y).sum()
            train_total += torch.numel(scores)
            train_cost_acum += cost.item()
            if mb%store_every == 0:
                val_cost, val_acc, dice, iou = accuracy(model, val_loader)
                train_acc = float(train_correct_num)/train_total
                train_cost_every = float(train_cost_acum)/mb
                print(f'mb: {mb}, train_cost: {train_cost_every:.3f}, val cost {val_cost:.3f},'
                        f'train acc: {train_acc:.4f}, val acc: {val_acc:.3f}, dice: {dice}, iou: {iou}')
                
                #saving data
                #train_acc_history.append(train_acc)
                #train_cost_history.append(train_cost_every)
        #train_acc = float(train_correct_num)/train_total
        #train_cost_every = float(train_cost_acum)/len(train_loader)
        #return train_acc_history
            


#Train model 

torch.manual_seed(42)
#imagenes RGB de entrada
#solamente 4 canales, no los 64 de la red original para reducir coste computacional
# 2 clases para segmentacion
epochs = 15
model = UNET(3,4,2)
criterion = nn.BCEWithLogitsLoss()
optimiser_unet = torch.optim.SGD(model.parameters(),
                                 lr = 0.01, momentum=0.95,
                                 weight_decay=1e-4
                                 )
#esta funcion se usa para encontrar el lr ideal. pero no se usa mas.
#la simulacion mostro que el valor optimo es 0.1
#lg_lr, losses, accuracies  = finding_lr(model, optimiser_unet, train_loader, start_val=1e-6, end_val=10)

scheduler = torch.optim.lr_scheduler.OneCycleLR(optimiser_unet,
                                                max_lr=0.1,
                                                steps_per_epoch=len(train_loader),
                                                epochs=epochs,pct_start=0.43,div_factor=10, final_div_factor=1000,
                                                three_phase=True
                                                )

train(model, optimiser_unet, scheduler, epochs)
