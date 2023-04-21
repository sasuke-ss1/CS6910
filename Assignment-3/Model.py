import torch
import torch.nn as nn
import sys

class Encoder(nn.Module):
    def __init__(self, inputSize: int, hiddenSize: int, numLayers: int, dropout: float, typ: str):
        super().__init__()
        self.hiddenSize = hiddenSize
        self.inputSize = inputSize
        self.numLayers = numLayers
        self.typ = typ.upper()

        self.embedding = nn.Embedding(inputSize, hiddenSize)
        

        if typ.upper() == "GRU":
            self.seq = nn.GRU(hiddenSize, hiddenSize, dropout=dropout, num_layers=numLayers)        
        elif typ.upper() == "LSTM":
            self.seq = nn.LSTM(hiddenSize, hiddenSize, dropout=dropout, num_layers=numLayers)
        elif typ.upper() == "RNN":
            self.seq = nn.RNN(hiddenSize, hiddenSize, dropout=dropout, num_layers=numLayers)
        else:
            raise NotImplementedError

    def forward(self, input, hidden):
        embed = self.embedding(input).view(1, 1, -1)
        output = embed

        output, hidden = self.seq(output, hidden)
                   
        return output, hidden
    
    def initHidden(self, device: torch.device):
        if self.typ == "LSTM":
            return (torch.rand(self.numLayers, 1, self.hiddenSize, device=device), torch.rand(self.numLayers, 1, self.hiddenSize, device=device))
        
        return torch.rand(self.numLayers, 1, self.hiddenSize, device=device)
    
class Decoder(nn.Module):
    def __init__(self, outputSize: int, hiddenSize: int, numLayers: int, dropout: float, typ: str):
        super().__init__()
        self.hiddenSize = hiddenSize
        self.outputSize = outputSize
        self.numLayers= numLayers
        self.typ = typ

        self.embedding = nn.Embedding(outputSize, hiddenSize)

        if typ.upper() == "GRU":
            self.seq = nn.GRU(hiddenSize, hiddenSize, dropout=dropout, num_layers=numLayers)        
        elif typ.upper() == "LSTM":
            self.seq = nn.LSTM(hiddenSize, hiddenSize, dropout=dropout, num_layers=numLayers)
        elif typ.upper() == "RNN":
            self.seq = nn.RNN(hiddenSize, hiddenSize, dropout=dropout, num_layers=numLayers)
        else:
            raise NotImplementedError

        self.out = nn.Linear(hiddenSize, outputSize)      
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        embed = self.embedding(input).view(1, 1, -1)
        output = embed
        output, hidden = self.seq(output, hidden)
        output = self.softmax(self.out(output[0]))
        
        return output, hidden
    
    def initHidden(self, device=None):
        return torch.rand(self.numLayers, 1, self.hiddenSize, device=device)