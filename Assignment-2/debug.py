import torch
import torch.nn as nn

input = torch.randn(64, 32, 2, 2)
output =nn.AdaptiveAvgPool2d((1,1))(input)
print(output.shape)