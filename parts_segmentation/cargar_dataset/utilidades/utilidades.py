import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def model_test(modelo):
    x = torch.randn((32,3,224,224))
    #model = modelo.UNET(3, 64, 2)
    preds = modelo(x)
    print("modelo evaluado")
    assert preds.shape != torch.Size([32,2,334,224]), "Hay algo mal en el modelo. Revisar"




def plot_mini_batch(train_loader, BATCH_SIZE: int, imgs, masks):
    ''' visualiza los graficos de un batch'''
    #imgs, masks = next(iter(train_loader))
    plt.figure(figsize=(20,10))
    for i in range(BATCH_SIZE):
        plt.subplot(4,8, i+1)
        img = imgs[i,...].permute(1,2,0).numpy()
        mask = masks[i,...].permute(1,2,0).numpy()
        plt.imshow(img)
        plt.imshow(mask, alpha=0.5)
        plt.axis('Off')
        plt.tight_layout()
    plt.show()



def accuracy(model, loader, criterion):
    correct = 0 
    intersection = 0
    denom = 0 
    union = 0 
    total = 0 
    cost = 0.
    model = model.to(device=device)
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype = torch.float32)            
            y = y.to(device = device, dtype = torch.long).squeeze(1)            
            #_, targety, targetx = y.shape
            scores  = model(x)
            #scores = torch.argmax(scores,dim=1).float()
            #batches,  height, width = scores.shape
            #start_Y = (height - targety) // 2
            #start_X = (width - targetx) // 2
            #scores =  scores[:,start_Y:start_Y + targety, start_X:start_X + targetx]
            cost += (criterion(scores, y)).item()
            #standard accuracy
            preds = torch.argmax(scores, dim=1)
            correct += (preds == y).sum()
            total += torch.numel(preds)
            #dice coefficient
            intersection += (preds*y).sum()
            denom +=  (preds + y).sum()
            dice = 2*intersection/(denom + 1e-8)
            #intersection over union
            union += (preds + y - preds*y).sum()
            iou  = (intersection)/(union + 1e-8)
        return cost/len(loader), float(correct/total), dice, iou
