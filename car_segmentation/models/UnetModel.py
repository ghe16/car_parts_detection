import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

class UNET(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, init_features=64, dropout_prob=0.5):
        super(UNET, self).__init__()

        features = init_features
        self.encoder1 = UNET._block(in_channels, features, dropout_prob)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNET._block(features, features * 2, dropout_prob)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNET._block(features * 2, features * 4, dropout_prob)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNET._block(features * 4, features * 8, dropout_prob)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNET._block(features * 8, features * 16, dropout_prob)

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNET._block((features * 8) * 2, features * 8, dropout_prob)
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNET._block((features * 4) * 2, features * 4, dropout_prob)
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNET._block((features * 2) * 2, features * 2, dropout_prob)
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNET._block(features * 2, features, dropout_prob)

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1, padding=(0,2)
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        enc4 = self.crop(enc4, dec4)  # Crop encoder feature map to match decoder
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        enc3 = self.crop(enc3, dec3)  # Crop encoder feature map to match decoder
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        enc2 = self.crop(enc2, dec2)  # Crop encoder feature map to match decoder
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        enc1 = self.crop(enc1, dec1)  # Crop encoder feature map to match decoder
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        return torch.sigmoid(self.conv(dec1))

    @staticmethod
    def crop(enc, dec):
        """Crop the encoder feature map to match the size of the decoder feature map."""
        _, _, h, w = dec.size()
        enc = TF.center_crop(enc, [h, w])
        return enc

    @staticmethod
    def _block(in_channels, features, dropout_prob):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob),
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob),
        )
