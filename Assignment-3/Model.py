import torch
import torch.nn as nn
import sys
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, inputSize: int, embedSize: int, hiddenSize: int, numLayers: int, dropout: float, bidirectional: bool,typ: str):
        super().__init__()
        self.hiddenSize = hiddenSize
        self.inputSize = inputSize
        self.numLayers = numLayers
        self.typ = typ.upper()
        self.bidirectional = bidirectional
        self.dropout = nn.Dropout(dropout)

        self.embedding = nn.Embedding(inputSize, embedSize)

        if typ.upper() == "GRU":
            self.seq = nn.GRU(embedSize, hiddenSize, dropout=dropout, num_layers=numLayers, bidirectional=bidirectional, batch_first=True)        
        elif typ.upper() == "LSTM":
            self.seq = nn.LSTM(embedSize, hiddenSize, dropout=dropout, num_layers=numLayers, bidirectional=bidirectional, batch_first=True)
        elif typ.upper() == "RNN":
            self.seq = nn.RNN(embedSize, hiddenSize, dropout=dropout, num_layers=numLayers, bidirectional=bidirectional, batch_first=True)
        else:
            raise NotImplementedError

    def forward(self, input, hidden):
        embed = self.embedding(input)
        embed = self.dropout(embed)
        
        output = embed

        output, hidden = self.seq(output, hidden)
                   
        return output, hidden
    
    def initHidden(self, batch_size: int, device: torch.device):
        if self.typ == "LSTM":
            return (torch.randn((1+int(self.bidirectional))*self.numLayers, batch_size, self.hiddenSize, device=device), torch.randn((1+int(self.bidirectional))*self.numLayers, batch_size, self.hiddenSize, device=device))
        
        return torch.randn((1+int(self.bidirectional))*self.numLayers, batch_size, self.hiddenSize, device=device)
    
class Decoder(nn.Module):
    def __init__(self, outputSize: int, embedSize: int,hiddenSize: int, numLayers: int, dropout: float, bidirectional: bool,typ: str):
        super().__init__()
        self.hiddenSize = hiddenSize
        self.outputSize = outputSize
        self.numLayers= numLayers
        self.typ = typ.upper()
        self.bidirectional = bidirectional
        self.dropout = nn.Dropout(dropout)

        self.embedding = nn.Embedding(outputSize, embedSize)

        if typ.upper() == "GRU":
            self.seq = nn.GRU(embedSize, hiddenSize, dropout=dropout, num_layers=numLayers, bidirectional=bidirectional, batch_first=True)        
        elif typ.upper() == "LSTM":
            self.seq = nn.LSTM(embedSize, hiddenSize, dropout=dropout, num_layers=numLayers, bidirectional=bidirectional, batch_first=True)
        elif typ.upper() == "RNN":
            self.seq = nn.RNN(embedSize, hiddenSize, dropout=dropout, num_layers=numLayers, bidirectional=bidirectional, batch_first=True)
        else:
            raise NotImplementedError

        self.out = nn.Linear((1+int(self.bidirectional))*self.hiddenSize, self.outputSize)      
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input: torch.Tensor, hidden: torch.Tensor):
        inp = input.unsqueeze(1)
        embed = self.embedding(inp)
        embed = self.dropout(embed)
        output = embed

        output, hidden = self.seq(output, hidden)

        output = self.softmax(self.out(output))

        return output, hidden
    
    def initHidden(self, batch_size: int,device: torch.device):
        if self.typ == "LSTM":
            return (torch.randn((1+int(self.bidirectional))*self.numLayers, batch_size, self.hiddenSize, device=device), torch.randn((1+int(self.bidirectional))*self.numLayers, batch_size, self.hiddenSize, device=device))
        
        return torch.randn((1+int(self.bidirectional))*self.numLayers, batch_size, self.hiddenSize, device=device)
    
class AttentionDecoder(nn.Module):
    def __init__(self, outputSize: int, embedSize: int,hiddenSize: int, numLayers: int, dropout: float, bidirectional: bool, maxLen: int,typ:str):
        super().__init__()
        self.hiddenSize = hiddenSize
        self.outputSize = outputSize
        self.numLayers = numLayers
        self.typ = typ.upper()
        self.maxLen = maxLen
        self.bidirectional = bidirectional

        self.embedding = nn.Embedding(outputSize, embedSize)
        
        self.attn = nn.Linear(2*hiddenSize*(1+int(bidirectional)),hiddenSize, bias=False)
        self.v = nn.Linear(hiddenSize,1, bias=False)


        if typ.upper() == "GRU":
            self.seq = nn.GRU((hiddenSize*(1+int(bidirectional))) + embedSize, hiddenSize, dropout=dropout, num_layers=numLayers, bidirectional=bidirectional, batch_first=True)        
        elif typ.upper() == "LSTM":
            self.seq = nn.LSTM((hiddenSize*(1+int(bidirectional))) + embedSize, hiddenSize, dropout=dropout, num_layers=numLayers, bidirectional=bidirectional, batch_first=True)
        elif typ.upper() == "RNN":
            self.seq = nn.RNN((hiddenSize*(1+int(bidirectional))) + embedSize, hiddenSize, dropout=dropout, num_layers=numLayers, bidirectional=bidirectional, batch_first=True)
        else:
            raise NotImplementedError
        
        self.out = nn.Linear((1+int(self.bidirectional))*self.hiddenSize, self.outputSize)     
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input: torch.Tensor, hidden: torch.Tensor, encoderOutputs):
        inp = input.unsqueeze(1)
        embed = self.embedding(inp).transpose(1, 0)
        s = hidden
        
        # Reshaping Stuff
        if self.typ == "LSTM":
            hidden = hidden[0].view(self.numLayers, 1+int(self.bidirectional), -1, self.hiddenSize)[-1]

            if self.bidirectional:
                h1, h2 = hidden[0], hidden[1]
                hidden = torch.cat((h1, h2), dim=1)

            else:
                hidden = hidden.squeeze(0)
        else:
            hidden = hidden.view(self.numLayers, 1+int(self.bidirectional), -1, self.hiddenSize)[-1]

            if self.bidirectional:
                h1, h2 = hidden[0], hidden[1]
                hidden = torch.cat((h1, h2), dim=1)

            else:
                hidden = hidden.squeeze(0)

        # Attention
        batchSize, seqLen, _ = encoderOutputs.shape
        hidden = hidden.unsqueeze(1).repeat(1, seqLen, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoderOutputs), dim = 2)))
        
        attention = self.v(energy).squeeze(2)
  
        attnWeights = F.softmax(attention, dim=1).unsqueeze(1)

        # Apply Attention
        attnApplied = torch.bmm(attnWeights, encoderOutputs).transpose(0, 1)
        input = torch.cat((embed, attnApplied), dim = 2).permute(1,0,2)

        output, hidden = self.seq(input, s)

        output = self.softmax(F.relu(self.out(output)))

        return output, hidden, attnWeights  

    def initHidden(self, batch_size: int,device: torch.device):
        if self.typ == "LSTM":
            return (torch.randn((1+int(self.bidirectional))*self.numLayers, batch_size, self.hiddenSize, device=device), torch.randn((1+int(self.bidirectional))*self.numLayers, batch_size, self.hiddenSize, device=device))
        
        return torch.randn((1+int(self.bidirectional))*self.numLayers, batch_size, self.hiddenSize, device=device)