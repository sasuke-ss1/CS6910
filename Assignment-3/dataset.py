import numpy as np
import pandas as pd
import os
import sys
import torch
from torch.utils.data import Dataset

np.random.seed(42)

class Dataset(Dataset):
    def __init__(self, path, batchSize=1,lang = "hin", sow="\t", eow="\n"):
        self.path = os.path.join(path, lang)
        self.path = list(map(lambda x: os.path.join(self.path, x), sorted(os.listdir(self.path))))
        self.sow = sow;self.eow=eow
        self.batchSize = batchSize

        self.data = self.readData()
        self.tokenize()


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
            self.maxXlen = max(self.maxXlen, len(x))
            for ch in x:
                xTok.add(ch)

            self.maxYlen = max(self.maxYlen, len(y))            
            for ch in y:
                yTok.add(ch)

        self.xTok, self.yTok = sorted(list(xTok)), sorted(list(yTok))

        self.xLen, self.yLen = len(self.xTok)+1, len(self.yTok)+1

        self.x2TDict = {ch:i+1 for i, ch in enumerate(self.xTok)}
        self.y2TDict = {ch:i+1 for i, ch in enumerate(self.yTok)}
        
        self.x2TDict[" "] = 0
        self.y2TDict[" "] = 0

        self.x2TDictR = {i: char for char, i in self.x2TDict.items()}
        self.y2TDictR = {i: char for char, i in self.y2TDict.items()}

    def _process(self, data: list):
        a = torch.zeros((len(data), self.maxXlen), dtype=torch.long)
        b = torch.zeros((len(data), self.maxYlen), dtype=torch.long)

        for i, [x, y] in enumerate(data):
            for j, ch in enumerate(x):
                a[i, j] = self.x2TDict[ch]
            a[i,j+1:] = self.x2TDict[" "]
            
            for j, ch in enumerate(y):
                b[i, j] = self.y2TDict[ch]
            b[i, j+1:] = self.y2TDict[" "]
            
    
        return a, b

    def get_data(self, name: str):
        if name.lower() == "train":
            return self._process(self.train)
        
        elif name.lower() == "test":
            return self._process(self.test)
        
        elif name.lower() == "val":
            return self._process(self.val)
        
        raise  NotImplementedError





if __name__ == "__main__":
    path = "aksharantar_sampled"
    data= Dataset(path)
    
    trainx, trainxx, trainy = data.get_data("train")
    print(trainx.shape, trainxx.shape, trainy.shape)
