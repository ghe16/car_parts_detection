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
from models.UnetModelMultiClass import *
from utilidades.utilidades import accuracy
from utilidades.classweights import calculate_class_weights


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

##Paths
PATH = "data/"
TRAIN_PATH = "data/train/JPEGImages/"
TRAIN_MASKS_PATH="data/train/JPEGMasks/"
TEST_PATH = ".data/test/"
#
BATCH_SIZE = 32

full_dataset = carga_carDataset.Car_Dataset(
    TRAIN_PATH,
    TRAIN_MASKS_PATH,
    data_augmented=False)
    

TRAIN_SIZE = int(len(full_dataset)*0.8)
VAL_SIZE = len(full_dataset) - TRAIN_SIZE

train_dataset, val_dataset = random_split(full_dataset, [TRAIN_SIZE,VAL_SIZE])

train_loader = DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=4)
val_loader   = DataLoader(val_dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=4)

#training

def train(model, optimiser, criterion, num_classes, epochs = 100, store_every = 10):
    print("entramos a entrenar")
    model = model.to(device= device)
    for epoch in range(epochs):
        model.train()
        train_correct_num = 0   # se resetea cada epoch
        train_total =  0   # num elementos evaluados
        train_cost_acum = 0. #  coste acumulado
        for mb , (x, y) in enumerate(train_loader, start=1):
            model.train()
            x= x.to(device=device)
            y = y.to(device=device)  # eliminamos el canal 1 (blanco y negro)
            #print(y.shape)
            #print(torch.max(y))
            scores  = model(x)
            #assert torch.max(y) < num_classes, "Label value exceeds number of classes"
            #assert torch.min(y) >= 0, "Label value is negative"

            #print( scores.shape[1])
            #print(scores.shape)
            cost = criterion(input=scores, target=y)
            optimiser.zero_grad()
            cost.backward()
            optimiser.step()
            train_predictions = torch.argmax(scores, dim = 1)
            train_correct_num += (train_predictions == y).sum()
            train_total += torch.numel(train_predictions)
            train_cost_acum += cost.item()
            if mb%store_every == 0:
                val_cost, val_acc, dice, iou = accuracy(model, val_loader,criterion, num_classes)
                train_acc = float(train_correct_num)/train_total
                train_cost_every = float(train_cost_acum)/mb
                print(f'mb: {mb}, train_cost: {train_cost_every:.3f}, val cost {val_cost:.3f},'
                        f'train acc: {train_acc:.4f}, val acc: {val_acc:.3f}, dice: {dice}, iou: {iou}')
                torch.save(model.state_dict(),'model_state_dict.pth')
                torch.save(model,'entire_model_multi.pth')            

def main():
    epochs = 100
    num_classes = 18
    dropout_prob = 0.5
    
    if os.path.isfile("entire_model_multi.pth"):
        model = torch.load("entire_model_multi.pth")
        #model = loaded_model.eval()
        print("existing model loaded")
    else:
        print("No se encontro el modelo")
        model = UNET(channels_in=3,out_channels=num_classes,init_features=4,dropout_prob=dropout_prob)

    optimiser_unet = torch.optim.Adam(model.parameters(),
                                    lr = 1e-3
                                    )
    #esta funcion se usa para encontrar el lr ideal. pero no se usa mas.
    #la simulacion mostro que el valor optimo es 0.1
    #lg_lr, losses, accuracies  = finding_lr(model, optimiser_unet, train_loader, start_val=1e-6, end_val=10)
    class_weights  = calculate_class_weights(TRAIN_MASKS_PATH,num_classes=num_classes)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights).float().to(device))
    #imagenes RGB de entrada
    #solamente 4 canales, no los 64 de la red original para reducir coste computacional
    # 2 clases para segmentacion


    train(model = model,optimiser = optimiser_unet, criterion = criterion,  num_classes = num_classes, epochs= epochs)

#Train model 
if __name__ == '__main__':
    main()
