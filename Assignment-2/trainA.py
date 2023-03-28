from Model import Model
from dataset import NatureData
import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision import transforms
from torch.utils.data import random_split, DataLoader
from tqdm import tqdm
import sys
import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parent_dir = "./nature_12K"
batch_size = 64
lr = 1e-3
betas = (0.9, 0.999)
epochs = 20

transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

data = NatureData(parent_dir, transforms=transform)
train_data, val_data = random_split(data, [8000, 1999])
test_data = NatureData(parent_dir, False, transform)

train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, num_workers=2)
val_loader = DataLoader(val_data, batch_size=batch_size)
test_loader = DataLoader(test_data, batch_size=batch_size)

loss_fn = nn.CrossEntropyLoss().to(device)
model = Model(num_filters=[3, 32, 32, 32, 32, 32], filter_size=3, pool_size=2, activation="ReLU")
model.to(device)
optim = Adam(model.parameters(), lr = lr, betas = betas)

def train():
    print("Training begins\n")
    
    for epoch in range(1, epochs+1):
        loop_obj = tqdm(train_loader)
        avg_loss = []
        for img, label in loop_obj:
            img, label = img.to(device), label.to(device)
            loop_obj.set_description(f"Epoch: {epoch}")
            optim.zero_grad()
            pred = model(img)
            loss = loss_fn(pred, label)
            loss.backward()
            optim.step()
            loop_obj.set_postfix_str(f"Loss: {loss.item():0.3f}")
            
            with torch.no_grad():
                avg_loss.append(loss.item())

        print(f"Average Loss in this Epoch: {sum(avg_loss)/len(avg_loss)}")
        print("Running Validation")

        with torch.no_grad():
            loop_obj = tqdm(val_loader)
            avg_acc = []        
            for img, label in loop_obj:
                img, label = img.to(device), label.to(device)
                pred = model(img)
                pred = torch.argmax(pred, dim=1)
                accuracy = (pred == label).float().mean()
                loop_obj.set_postfix_str(f"Accuracy: {accuracy:0.3f}")
                avg_acc.append(accuracy)
            print(f"Validation Accuracy in this Epoch: {sum(avg_acc)/len(avg_acc)}")


if __name__ == "__main__":
    train()