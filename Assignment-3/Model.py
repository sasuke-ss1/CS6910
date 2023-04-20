import torch
import torch.nn as nn
import sys

class Encoder(nn.Module):
    def __init__(self, inputSize: int, hiddenSize: int, typ: str):
        super().__init__()
        self.hiddenSize = hiddenSize
        self.inputSize = inputSize
        self.typ = typ

        self.embedding = nn.Embedding(inputSize, hiddenSize)
        if typ.upper() == "GRU":
            self.seq = nn.GRU(hiddenSize, hiddenSize)        
        
    def forward(self, input, hidden):
        embed = self.embedding(input).view(1, 1, -1)
        output = embed
        output, hidden = self.seq(output, hidden)
                   
        return output, hidden
    
    def initHidden(self, device: torch.device):
        return torch.zeros(1, 1, self.hiddenSize, device=device)
    
class Decoder(nn.Module):
    def __init__(self, outputSize: int, hiddenSize: int, typ: str):
        super().__init__()
        self.hiddenSize = hiddenSize
        self.outputSize = outputSize
        self.typ = typ

        self.embedding = nn.Embedding(outputSize, hiddenSize)
        if typ.upper() == "GRU":
            self.seq = nn.GRU(hiddenSize, hiddenSize)

        self.out = nn.Linear(hiddenSize, outputSize)      
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        embed = self.embedding(input).view(1, 1, -1)
        output = embed
        output, hidden = self.seq(output, hidden)
        output = self.softmax(self.out(output[0]))
        
        return output, hidden
    
    def initHidden(self):
        return torch.zeros(1, 1, self.hiddenSize)