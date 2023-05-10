import numpy as np
import pandas as pd
import os
import sys
import torch
from torch.utils.data import Dataset, DataLoader


class dataset(Dataset):
    def __init__(self, path, batchSize=1,lang = "hin", typ="train",sow="\t", eow="\n"):
        self.path = os.path.join(path, lang)
        self.path = list(map(lambda x: os.path.join(self.path, x), sorted(os.listdir(self.path))))
        self.sow = sow;self.eow=eow
        self.batchSize = batchSize
        self.typ = typ.lower()

        self.datx, self.daty = None, None

        self.data = self.readData()
        self.tokenize()
        self.get_data()

    @staticmethod
    def _readData(path: str, sow, eow):
        with open(path, "r") as f:
            lines = [line.split(",") for line in f.read().split("\n") if line != '']
            
        lines = [[sow + s + eow for s in line] for line in lines]
        return lines

    def readData(self):
        self.test, self.train, self.val = [self._readData(i, self.sow, self.eow) for i in self.path]

    def tokenize(self):
        xTok, yTok = set(), set()
        self.maxXlen, self.maxYlen = -1, -1

        for x, y in self.train:
            for ch in x:
                xTok.add(ch)
         
            for ch in y:
                yTok.add(ch)

        self.xTok, self.yTok = sorted(list(xTok)), sorted(list(yTok))

        self.xLen, self.yLen = len(self.xTok)+2, len(self.yTok)+2

        self.x2TDict = {ch:i+2 for i, ch in enumerate(self.xTok)}
        self.y2TDict = {ch:i+2 for i, ch in enumerate(self.yTok)}
        
        self.x2TDict[" "] = 0
        self.y2TDict[" "] = 0

        self.x2TDict["unk"] = 1
        self.y2TDict["unk"] = 1

        self.x2TDictR = {i: char for char, i in self.x2TDict.items()}
        self.y2TDictR = {i: char for char, i in self.y2TDict.items()}

    def _process(self, data: list):
        for (x, y) in data:
            self.maxXlen = max(self.maxXlen, len(x))
            self.maxYlen = max(self.maxYlen, len(y)) 

        a = torch.zeros((len(data), self.maxXlen), dtype=torch.long)
        b = torch.zeros((len(data), self.maxYlen), dtype=torch.long)

        for i, [x, y] in enumerate(data):
            for j, ch in enumerate(x):
                if ch in self.x2TDict.keys():
                    a[i, j] = self.x2TDict[ch]
                else:
                    a[i, j] = self.x2TDict["unk"]

            a[i,j+1:] = self.x2TDict[" "]
            
            for j, ch in enumerate(y):
                if ch in self.y2TDict.keys():
                    b[i, j] = self.y2TDict[ch]
                else:
                    b[i, j] = self.y2TDict["unk"]

            b[i, j+1:] = self.y2TDict[" "]
            
        
        return a, b

    def get_data(self):
        if self.typ == "train":
            self.datx, self.daty = self._process(self.train)
        
        elif self.typ == "test":
            self.datx, self.daty = self._process(self.test)
        
        elif self.typ == "val":
            self.datx, self.daty = self._process(self.val)
        
        else:
            raise  NotImplementedError
    

    def __len__(self):
        if self.typ == "train":
            return len(self.train)
        
        elif self.typ == "val":
            return len(self.val)
        
        elif self.typ == "test":
            return len(self.test)
    
    def __getitem__(self, index) -> torch.Tensor:
        return self.datx[index], self.daty[index]



if __name__ == "__main__":
    path = "aksharantar_sampled"
    data= dataset(path, typ="test")
    print(data.__len__())
    dataloader = DataLoader(data, batch_size=32)

    print(data.xLen, data.yLen)

    for trainx, trainy in dataloader:
        print(trainx.shape)
        print(trainy.shape)
    
        sys.exit()    

            