from torch.nn import functional as F
from torch import nn, optim
import torch 


class Conv_3_k(nn.Module):
    def __init__(self,channels_in,channels_out):
        super().__init__()
        self.conv1 = nn.Conv2d(channels_in,channels_out,kernel_size=3,stride=1,padding=1)

    def forward(self,x):
        return self.conv1(x)

class Double_Conv(nn.Module):
    '''Capas dobles de convoluciones para downsampling'''
    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.double_conv = nn.Sequential(
                            Conv_3_k(channels_in,channels_out),
                            nn.BatchNorm2d(channels_out),
                            nn.ReLU(),

                            Conv_3_k(channels_out,channels_out),
                            nn.BatchNorm2d(channels_out),
                            nn.ReLU()
        )
    def forward(self, x):
        return self.double_conv(x)

class Down_conv(nn.Module):
    '''
    Downsampling
    '''
    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.MaxPool2d(2,2),
            Double_Conv(channels_in,channels_out)
        )

    def forward(self,x):
        return self.encoder(x)
    

class Up_conv(nn.Module):
    """Proceso inverson de subida. En vez de usar transpose conv layers, usamos upsample + 1x1 conv"""
    def __init__(self,channels_in,channels_out):
        super().__init__()
        self.upsample_layer = nn.Sequential(
                                nn.Upsample(scale_factor=2, mode='bilinear'),
                                nn.Conv2d(channels_in,channels_in//2, kernel_size=1,stride=1,padding=1)
                                )
        self.decoder = Double_Conv(channels_in,channels_out)

    def forward(self,x1,x2):
        """x1 es l feature map que viene de abajo 
        x2 es el feature map que viene de la parte de downsampling (espejo)"""        
        x1 = self.upsample_layer(x1)
        x1 = x1[:,:,0:x2.size()[2],0: x2.size()[3]]
        x = torch.cat([x2,x1],dim=1)
        return self.decoder(x)


class UNET(nn.Module):
    """modelo total integrado"""
    def __init__(self, channels_in, channels, num_classes):
        super().__init__()
        self.first_conv = Double_Conv(channels_in, channels)      # 64   224 x 224
        self.down_conv1 = Down_conv(channels    , channels * 2)   # 128  112 x 112
        self.down_conv2 = Down_conv(channels * 2, channels * 4)   # 256  56 X 56
        self.down_conv3 = Down_conv(channels * 4, channels * 8)   # 512  28 X 28
        self.middle_conv = Down_conv(channels * 8, channels * 16)  # 1024 14 x 14

        self.up_conv1 = Up_conv(channels * 16, channels * 8)  # 1024 52 X52
        self.up_conv2 = Up_conv(channels *  8, channels * 4)
        self.up_conv3 = Up_conv(channels *  4, channels * 2)
        self.up_conv4 = Up_conv(channels *  2, channels  )

        self.last_conv = nn.Conv2d(channels, num_classes, kernel_size=1, stride= 1,padding =1)

    def forward(self,x):
        x1 = self.first_conv(x)
        x2 = self.down_conv1(x1)
        x3 = self.down_conv2(x2)
        x4 = self.down_conv3(x3)

        x5 = self.middle_conv(x4)

        u1 = self.up_conv1(x5,x4)
        u2 = self.up_conv2(u1,x3)
        u3 = self.up_conv3(u2,x2)
        u4 = self.up_conv4(u3,x1)

        return self.last_conv(u4)
