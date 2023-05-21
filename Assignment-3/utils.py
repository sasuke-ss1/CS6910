import torch
import pandas as pd
import numpy as  np
import matplotlib.pyplot as plt

def wordAccuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred1 = torch.argmax(pred, dim = 1)
    bs = target.shape[0]

    with torch.no_grad():
        c = 0
        for i in range(bs):
            if ((pred1[i] == target[i]).sum().item() == target.shape[1]):
                c += 1
    
    return c/bs

def charAccuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
    numChar = pred.shape[1]
    bs = pred.shape[0]

    with torch.no_grad():
        pred = torch.argmax(pred, dim = 1)
        c = (pred == target).sum().item()
  
    return c/(numChar*bs)

def word2csv(pred: torch.Tensor, invDict: dict, path: str, testPath: str):
    batchSize, len = pred.shape
    letters = np.zeros((batchSize, len-1), str)
    words = []

    for i in range(1, len):
        p = pred[:, i]
        p = np.array([invDict[i.item()] for i in p])
        letters[:, i-1] = p
        

    
    words = [("".join(letters[i, :]).replace(" ", "")) for i in range(batchSize)]
    column_names=["input","true"]
    df = pd.read_csv(testPath, header=None, names=column_names)
    df["prediction"] = words

    df.to_csv(path + ".csv")
    
def plot(attn: torch.Tensor, ins: torch.Tensor, tars: torch.Tensor, int2char: dict,name: str):
    fig, axs = plt.subplots(3, 3, figsize=(10, 20))

    for i in range(9):
        axs[i//3, i%3].imshow(attn[i, :].cpu().numpy().T)
        axs[i//3, i%3].set_yticks([])
        axs[i//3, i%3].set_xticks([])
                                  
        axs[i//3, i%3].set_title("".join([int2char[1][j.item()] for j in ins[i]])[1:])
        

    plt.savefig(name + ".png")

if __name__ == "__main__":
    df = pd.read_csv("pred.csv", index_col=0)
    print(df.head())

