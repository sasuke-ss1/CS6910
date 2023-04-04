import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50

class Model(nn.Module):
    def __init__(self, num_filters, filter_size, pool_size, activation, img_shape=224, dropout=None, batch_norm=False, num_classes=10):
        super().__init__()
        if len(filter_size) == 1:
            filter_size = 5*[filter_size[0]]
        layers = [];dim = img_shape
        for i in range(5):
            layers.append(convActPool(num_filters[i], num_filters[i+1], filter_size[i], pool_size, activation))
            dim = dim + 2-(filter_size[i]-1)
            dim = (dim - (2-1)-1)//2 + 1
        
        layers.append(nn.Flatten())    
        layers.append(nn.Linear(dim*dim*num_filters[-1], 10))
        
        self.logits = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.logits(x)
        return logits


class convActPool(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, pool_size, activation, dropout=None, batch_norm=False):
        super().__init__()
        conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, padding=1)
        activation = getattr(nn, activation)()
        pool = nn.MaxPool2d(pool_size)
        layers = [conv, activation, pool]

        if dropout:
            layers.append(nn.Dropout(dropout))
        
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channel))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        out = self.network(x)
        
        return out


def get_resnet(num_classes, fineTune = True):
    ResNet = resnet50(pretrained=True)
    if fineTune:
        for param in ResNet.parameters():
            param.requires_grad = False

    n_inputs = ResNet.fc.in_features

    ResNet.fc = nn.Sequential(
        nn.Linear(n_inputs,2048),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(2048, num_classes)
        )
                
        #return ResNet
    
    return ResNet


class ResNet_18(nn.Module):
    
    def __init__(self, image_channels, num_classes):
        
        super(ResNet_18, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        #resnet layers
        self.layer1 = self.__make_layer(64, 64, stride=1)
        self.layer2 = self.__make_layer(64, 128, stride=2)
        self.layer3 = self.__make_layer(128, 256, stride=2)
        self.layer4 = self.__make_layer(256, 512, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        
    def __make_layer(self, in_channels, out_channels, stride):
        
        identity_downsample = None
        if stride != 1:
            identity_downsample = self.identity_downsample(in_channels, out_channels)
            
        return nn.Sequential(
            Block(in_channels, out_channels, identity_downsample=identity_downsample, stride=stride), 
            Block(out_channels, out_channels)
        )
        
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x 
    
    def identity_downsample(self, in_channels, out_channels):
        
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1), 
            nn.BatchNorm2d(out_channels)
        )
    

class Block(nn.Module):
    
    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample
        
    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
        x += identity
        x = self.relu(x)
        return x
