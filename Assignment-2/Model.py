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
                
    
    return ResNet

if  __name__ == "__main__":
    ResNet = get_resnet(10)