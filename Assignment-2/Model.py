import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, num_filters, filter_size, pool_size, activation, img_shape=150):
        super().__init__()
        layers = [];dim = img_shape
        for i in range(5):
            layers.append(convActPool(num_filters[i], num_filters[i+1], filter_size, pool_size,activation))
            dim = (dim - (pool_size - 1) - 1)//pool_size + 1
        layers.append(nn.Flatten())
        layers.append(nn.Linear(dim*dim*num_filters[-1], 10))
        
        self.logits = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.logits(x)
        return logits


class convActPool(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, pool_size, activation):
        super().__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, padding="same")
        self.activation = getattr(nn, activation)()
        self.pool = nn.MaxPool2d(pool_size)

    def forward(self, x):
        out = self.conv(x)
        out = self.activation(out)
        out = self.pool(out)
    
        return out
