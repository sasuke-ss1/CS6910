import numpy as np
import pandas as pd
import os

END_CHAR = "\n"
SEP = ","


def read_data(path: str):
    with open(path, "r") as f:
        lines = [line.split(",") for line in f.read().split("\n") if line != '']
        
    return lines

def encode(lines: list, train=False, x2enc=None, y2enc=None):
    if train:
        xChar, yChar = set(), set()
        maxXLen, maxYLen = -1, -1
        for x, y in lines:
            maxXLen = max(maxXLen, len(x))
            for char in x:
                xChar.add(char)
            
            maxYLen = max(maxYLen, len(y))
            
            for char in y:
                yChar.add(char)    

        
        x2enc = {char: i for i, char in enumerate(xChar)}
        y2enc = {char: i for i, char in enumerate(yChar)}
        x2enc["maxXlen"] = maxXLen
        y2enc["maxYlen"] = maxYLen

    xEnc = np.zeros((len(lines), x2enc["maxXlen"], len(x2enc.items())), np.float32)    
    yEnc = np.zeros((len(lines), y2enc["maxYlen"], len(y2enc.items())), np.float32)

    for i, [x, y] in enumerate(lines):
        for idx, char in enumerate(x):
            xEnc[i, idx, x2enc[char]] = 1

        for idx, char in enumerate(y):
            yEnc[i, idx, y2enc[char]] = 1

    if train:
        return [xEnc, yEnc], [x2enc, y2enc] 

    return [xEnc, yEnc]





def load_data(path: str):
    files = list(map(lambda x: os.path.join(path, x), sorted(os.listdir(path))))
    test, train, val =list(map(lambda x: read_data(x), files))

    train, [x2enc, y2enc] = encode(train, True)
    val = encode(val, x2enc=x2enc, y2enc=y2enc)#, encode(test, x2enc=x2enc, y2enc=y2enc)

    #print(test[0].shape, test[1].shape)

    return train, val, test







if __name__ == "__main__":
    path = "aksharantar_sampled/hin"
    train, val, test = load_data(path)
    print(train[0].shape)
    print(val[0].shape)