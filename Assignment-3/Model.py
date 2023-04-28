import torch
import torch.nn as nn
import sys
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, inputSize: int, hiddenSize: int, numLayers: int, dropout: float, bidirectional: bool,typ: str):
        super().__init__()
        self.hiddenSize = hiddenSize
        self.inputSize = inputSize
        self.numLayers = numLayers
        self.typ = typ.upper()
        self.bidirectional = bidirectional

        self.embedding = nn.Embedding(inputSize, hiddenSize)
        

        if typ.upper() == "GRU":
            self.seq = nn.GRU(hiddenSize, hiddenSize, dropout=dropout, num_layers=numLayers, bidirectional=bidirectional, batch_first=True)        
        elif typ.upper() == "LSTM":
            self.seq = nn.LSTM(hiddenSize, hiddenSize, dropout=dropout, num_layers=numLayers, bidirectional=bidirectional, batch_first=True)
        elif typ.upper() == "RNN":
            self.seq = nn.RNN(hiddenSize, hiddenSize, dropout=dropout, num_layers=numLayers, bidirectional=bidirectional, batch_first=True)
        else:
            raise NotImplementedError

    def forward(self, input, hidden):
        embed = self.embedding(input)

        output = embed

        output, hidden = self.seq(output, hidden)
                   
        return output, hidden
    
    def initHidden(self, batch_size: int, device: torch.device):
        if self.typ == "LSTM":
            return (torch.randn((1+int(self.bidirectional))*self.numLayers, batch_size, self.hiddenSize, device=device), torch.randn((1+int(self.bidirectional))*self.numLayers, batch_size, self.hiddenSize, device=device))
        
        return torch.randn((1+int(self.bidirectional))*self.numLayers, batch_size, self.hiddenSize, device=device)
    
class Decoder(nn.Module):
    def __init__(self, outputSize: int, hiddenSize: int, numLayers: int, dropout: float, bidirectional: bool,typ: str):
        super().__init__()
        self.hiddenSize = hiddenSize
        self.outputSize = outputSize
        self.numLayers= numLayers
        self.typ = typ.upper()
        self.bidirectional = bidirectional

        self.embedding = nn.Embedding(outputSize, hiddenSize)

        if typ.upper() == "GRU":
            self.seq = nn.GRU(hiddenSize, hiddenSize, dropout=dropout, num_layers=numLayers, bidirectional=bidirectional, batch_first=True)        
        elif typ.upper() == "LSTM":
            self.seq = nn.LSTM(hiddenSize, hiddenSize, dropout=dropout, num_layers=numLayers, bidirectional=bidirectional, batch_first=True)
        elif typ.upper() == "RNN":
            self.seq = nn.RNN(hiddenSize, hiddenSize, dropout=dropout, num_layers=numLayers, bidirectional=bidirectional, batch_first=True)
        else:
            raise NotImplementedError

        self.out = nn.Linear((1+int(self.bidirectional))*self.hiddenSize, self.outputSize)      
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input: torch.Tensor, hidden: torch.Tensor):
        inp = input.unsqueeze(1)
        embed = self.embedding(inp)
        output = embed

        output, hidden = self.seq(output, hidden)

        output = self.softmax(self.out(output))
        
        return output, hidden
    
    def initHidden(self, batch_size: int,device: torch.device):
        if self.typ == "LSTM":
            return (torch.randn((1+int(self.bidirectional))*self.numLayers, batch_size, self.hiddenSize, device=device), torch.randn((1+int(self.bidirectional))*self.numLayers, batch_size, self.hiddenSize, device=device))
        
        return torch.randn((1+int(self.bidirectional))*self.numLayers, batch_size, self.hiddenSize, device=device)
    
class AttentionDecoder(nn.Module):
    def __init__(self, outputSize: int, hiddenSize: int, numLayers: int, dropout: float, bidirectional: bool, maxLen: int,typ:str):
        super().__init__()
        self.hiddenSize = hiddenSize
        self.outputSize = outputSize
        self.numLayers = numLayers
        self.typ = typ.upper()
        self.maxLen = maxLen

        self.embedding = nn.Embedding(outputSize, hiddenSize)
        self.attn = nn.Linear(hiddenSize*2, maxLen)
        self.attnCombine = nn.Linear(hiddenSize*2, hiddenSize)

        if typ.upper() == "GRU":
            self.seq = nn.GRU(hiddenSize, hiddenSize, dropout=dropout, num_layers=numLayers, bidirectional=bidirectional)        
        elif typ.upper() == "LSTM":
            self.seq = nn.LSTM(hiddenSize, hiddenSize, dropout=dropout, num_layers=numLayers, bidirectional=bidirectional)
        elif typ.upper() == "RNN":
            self.seq = nn.RNN(hiddenSize, hiddenSize, dropout=dropout, num_layers=numLayers, bidirectional=bidirectional)
        else:
            raise NotImplementedError

        self.out = nn.Linear(hiddenSize, outputSize)

    def forward(self, input, hidden, encoderOutputs):
        embed = self.embedding(input).view(1, 1, -1)
        if self.typ == "LSTM":
            attnWeights = F.softmax(self.attn(torch.cat((embed[0], hidden[0][0]), 1)), dim=1)
        else:
            attnWeights = F.softmax(self.attn(torch.cat((embed[0], hidden[0]), 1)), dim=1)
        attnApplied = torch.bmm(attnWeights.unsqueeze(0), encoderOutputs.unsqueeze(0))

        output = torch.cat((embed[0], attnApplied[0]), 1)
        output = self.attnCombine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.seq(output, hidden)
        output = F.log_softmax(self.out(output[0]), dim=1)

        return output, hidden, attnWeights  

    def initHidden(self, batch_size, device: torch.device):
        if self.typ == "LSTM":
            return (torch.rand(self.numLayers, batch_size, self.hiddenSize, device=device), torch.rand(self.numLayers, batch_size, self.hiddenSize, device=device))
        
        return torch.rand(self.numLayers, batch_size, self.hiddenSize, device=device)