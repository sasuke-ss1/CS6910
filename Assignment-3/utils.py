import torch


def wordAccuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred = torch.argmax(pred, dim = 1)
    bs = target.shape[0]

    with torch.no_grad():
        c = 0
        for i in range(bs):
            if ((pred[i,:] == target[i,:]).sum().item() == target.shape[1]):
                c += 1
    
    return c/bs

def charAccuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
    numChar = pred.shape[1]
    bs = pred.shape[0]

    with torch.no_grad():
        pred = torch.argmax(pred, dim = 1)
        c = (pred == target).sum().item()
  
    return c/(numChar*bs)