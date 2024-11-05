import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import timm

def double_conv(in_channels, out_channels, dropout=0.0):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    ]
    if dropout > 0.0:
        layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)

class UNet(nn.Module):
    def __init__(self, dropout_rate=0.2):
        super(UNet, self).__init__()
        self.conv_down1 = double_conv(3, 64)
        self.conv_down2 = double_conv(64, 128)
        self.conv_down3 = double_conv(128, 256)
        self.conv_down4 = double_conv(256, 512, dropout=dropout_rate)

        self.bottleneck = double_conv(512, 1024, dropout=dropout_rate)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_up3 = double_conv(1024 + 256, 256, dropout=dropout_rate)
        self.conv_up2 = double_conv(256 + 128, 128)
        self.conv_up1 = double_conv(128 + 64, 64)
        self.last_conv = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        conv1 = self.conv_down1(x)
        x = self.maxpool(conv1)
        conv2 = self.conv_down2(x)
        x = self.maxpool(conv2)
        conv3 = self.conv_down3(x)
        x = self.maxpool(conv3)
        conv4 = self.conv_down4(x)

        bottleneck = self.bottleneck(conv4)
        x = self.upsample(bottleneck)

        x = torch.cat([x, conv3], dim=1)
        x = self.conv_up3(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)
        x = self.conv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)
        x = self.conv_up1(x)
        out = self.last_conv(x)
        out = torch.sigmoid(out)
        return out
    
#Models for features extraction
def load_model(model_name='resnet', freeze_layers=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if model_name == 'resnet':
        model = models.resnet50(weights="IMAGENET1K_V1")
        model = nn.Sequential(*list(model.children())[:-1])
        if freeze_layers:
            for param in model.parameters():
                param.requires_grad = False
    elif model_name == 'densenet':
        model = models.densenet121(weights='IMAGENET1K_V1')
        model = nn.Sequential(*list(model.children())[:-1], nn.AdaptiveAvgPool2d((1, 1)))
        if freeze_layers:
            for param in model.parameters():
                param.requires_grad = False
    elif model_name == 'vit':
        model = timm.create_model('vit_base_patch16_224', pretrained=True)
        model.head = nn.Identity()  # Remove the classification head
        if freeze_layers:
            for param in model.parameters():
                param.requires_grad = False
    else:
        raise ValueError("Invalid model name. Choose either 'resnet', 'densenet', or 'vit'.")
    
    model = model.to(device)
    
    return model, device