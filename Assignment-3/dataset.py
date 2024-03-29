import numpy as np
import pandas as pd
import os
import sys
import torch
from torch.utils.data import Dataset, DataLoader
from utils import word2csv


class dataset(Dataset):
    '''
    This class houses all the three types of data i.e. the Train, Test and Val data
    '''
    def __init__(self, path, batchSize=1,lang = "hin", typ="train",sow="\t", eow="\n"):
        self.path = os.path.join(path, lang)
        self.path = list(map(lambda x: os.path.join(self.path, x), sorted(os.listdir(self.path)))) # Set the paths
        self.sow = sow;self.eow=eow # Set EOW and SOW
        self.batchSize = batchSize # Get batch size(used for testing only)
        self.typ = typ.lower() # Get the type of data required finally

        self.datx, self.daty = None, None # Placeholder variables

        self.data = self.readData() # Reads the data as it is
        self.tokenize() # Tokenizes the charecters
        self.get_data() # Set the data to the placeholder variables

    @staticmethod
    def _readData(path: str, sow, eow):
        # Internal function for loading each lines of the csv file
        with open(path, "r") as f:
            lines = [line.split(",") for line in f.read().split("\n") if line != '']
            
        lines = [[sow + s + eow for s in line] for line in lines]
        return lines

    def readData(self):
        # Public function calling the _readData function for the data
        self.test, self.train, self.val = [self._readData(i, self.sow, self.eow) for i in self.path]

    def tokenize(self):
        # Tokenization
        xTok, yTok = set(), set()
        self.maxXlen, self.maxYlen = -1, -1

        # For each word in the train data we add assign them token in increasing order
        for x, y in self.train:
            for ch in x:
                xTok.add(ch)
         
            for ch in y:
                yTok.add(ch)

        self.xTok, self.yTok = sorted(list(xTok)), sorted(list(yTok))

        self.xLen, self.yLen = len(self.xTok)+2, len(self.yTok)+2

        # Create character to index mapping.
        self.x2TDict = {ch:i+2 for i, ch in enumerate(self.xTok)}
        self.y2TDict = {ch:i+2 for i, ch in enumerate(self.yTok)}
        
        # 0 index is for pad token 
        self.x2TDict[" "] = 0 
        self.y2TDict[" "] = 0

        # 1 index is for the unknown token
        self.x2TDict["\r"] = 1 
        self.y2TDict["\r"] = 1

        # Create a reverse mapping from index to character
        self.x2TDictR = {i: char for char, i in self.x2TDict.items()}
        self.y2TDictR = {i: char for char, i in self.y2TDict.items()}

    def _process(self, data: list):
        # Internal function for processing the data and creating tensors.
        for (x, y) in data:
            self.maxXlen = max(self.maxXlen, len(x))
            self.maxYlen = max(self.maxYlen, len(y)) 

        # Placeholder tensor variables
        a = torch.zeros((len(data), self.maxXlen), dtype=torch.long) 
        b = torch.zeros((len(data), self.maxYlen), dtype=torch.long)

        # Setting the values to the placeholder tensors
        for i, [x, y] in enumerate(data):
            for j, ch in enumerate(x):
                if ch in self.x2TDict.keys():
                    a[i, j] = self.x2TDict[ch]
                else:
                    a[i, j] = self.x2TDict["\r"]

            a[i,j+1:] = self.x2TDict[" "]
            
            for j, ch in enumerate(y):
                if ch in self.y2TDict.keys():
                    b[i, j] = self.y2TDict[ch]
                else:
                    b[i, j] = self.y2TDict["\r"]

            b[i, j+1:] = self.y2TDict[" "]
            
        
        return a, b

    def get_data(self):
        # Set the data in the placeholder variables
        if self.typ == "train":
            self.datx, self.daty = self._process(self.train)
        
        elif self.typ == "test":
            self.datx, self.daty = self._process(self.test)
        
        elif self.typ == "val":
            self.datx, self.daty = self._process(self.val)
        
        else:
            raise  NotImplementedError
    

    def __len__(self):
        # Returns the length of the dataset
        if self.typ == "train":
            return len(self.train)
        
        elif self.typ == "val":
            return len(self.val)
        
        elif self.typ == "test":
            return len(self.test)
    
    def __getitem__(self, index) -> torch.Tensor:
        # GetItem overwrite for custom torch dataset
        return self.datx[index], self.daty[index]



if __name__ == "__main__":
    path = "aksharantar_sampled"
    data= dataset(path, typ="test")
    
    loader = DataLoader(data, batch_size=32)

    for pair in loader:

        word2csv(pair[1], data.y2TDictR, "./1")
        break


            