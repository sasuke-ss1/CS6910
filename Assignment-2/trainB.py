from Model import get_resnet
from dataset import NatureData
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import sys


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parent_dir = "./nature_12K"
batch_size = 128
lr = 1e-3
epochs = 20

train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomApply(torch.nn.ModuleList([transforms.ColorJitter()]), p=0.1),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.RandomErasing(p=0.2, value='random')

])

val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

data = NatureData(parent_dir, transforms=train_transform)
train_data, val_data = random_split(data, [8000, 1999])
test_data = NatureData(parent_dir, False, val_transform)

train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, num_workers=2)
val_loader = DataLoader(val_data, batch_size=batch_size)
test_loader = DataLoader(test_data, batch_size=batch_size)

train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, num_workers=2)
val_loader = DataLoader(val_data, batch_size=batch_size)
test_loader = DataLoader(test_data, batch_size=batch_size)

loss_fn = CrossEntropyLoss()
ResNet = get_resnet(10)
optim = Adam(ResNet.parameters(), lr=lr)


def train():
    torch.backends.cudnn.benchmark = True
    print("Training begins\n")
    ResNet.to(device)

    for epoch in range(1, epochs+1):
        loop_obj = tqdm(train_loader)
        avg_loss, avg_acc = [], []
        ResNet.train()
        for img, label in loop_obj:
            img, label = img.to(device), label.to(device)
            loop_obj.set_description(f"Epoch: {epoch}")
            optim.zero_grad()
            pred = ResNet(img)
            loss = loss_fn(pred, label)
            loss.backward()
            optim.step()
            
            with torch.no_grad():
                avg_loss.append(loss.item())
                accuracy = (torch.argmax(pred, dim=1) == label).float().mean()
                avg_acc.append(accuracy)
                loop_obj.set_postfix_str(f"Loss: {loss.item():0.3f}, Accuracy: {accuracy:0.3f}")

        print(f"Average Loss in this Epoch: {sum(avg_loss)/len(avg_loss)}")
        print(f"Average Accuracy in this Epoch: {sum(avg_acc)/len(avg_acc)}")
        print("Running Validation")

        with torch.no_grad():
            loop_obj = tqdm(val_loader)
            avg_acc = []
            ResNet.eval()        
            for img, label in loop_obj:
                img, label = img.to(device), label.to(device)
                pred = ResNet(img)
                pred = torch.argmax(pred, dim=1)
                accuracy = (pred == label).float().mean()
                loop_obj.set_postfix_str(f"Accuracy: {accuracy:0.3f}")
                avg_acc.append(accuracy)
            print(f"Validation Accuracy in this Epoch: {sum(avg_acc)/len(avg_acc)}")


if __name__ == "__main__":
    train()
