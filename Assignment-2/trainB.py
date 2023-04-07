from Model import get_resnet
from dataset import NatureData
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import sys
from argparse import ArgumentParser
import wandb
import yaml

parser = ArgumentParser()

parser.add_argument("--wandb_project", "-wp", default="Asng-2", type=str, help="Project name used to track experiments in Weights & Biases dashboard")
parser.add_argument("--wandb_entity", "-we", default="sasuke", type=str, help="Wandb Entity used to track experiments in the Weights & Biases dashboard.")
parser.add_argument("--batch_size", "-b", default=64, type=int, help="Batch size used to train neural network.")
parser.add_argument("--question", "-q", type=bool, default=False, help="Set True to run wandb experiments")
parser.add_argument("--lr", "-lr", type=float, default=1e-5, help="Learning rate used to optimize model parameters")
parser.add_argument("--epochs", "-e", default=6, type=int, help="Number of epochs to train neural network.")
parser.add_argument("--parent_dir", "-p", type=str, default="./nature_12K", help="Path to the parent directory of the dataset.")

args = parser.parse_args()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parent_dir = args.parent_dir
batch_size = args.batch_size
lr = args.lr
epochs = args.epochs

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

train_data = NatureData(parent_dir, transforms=train_transform)
test_data = NatureData(parent_dir, False, val_transform)

train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, num_workers=2)
test_loader = DataLoader(test_data, batch_size=batch_size)

train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, num_workers=2)
test_loader = DataLoader(test_data, batch_size=batch_size)

loss_fn = CrossEntropyLoss()
ResNet = get_resnet(10)



def train():
    optim = Adam(ResNet.parameters(), lr=lr)
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
            loop_obj = tqdm(test_loader)
            val_avg_loss, val_avg_acc = [], []        
            ResNet.eval()
            for img, label in loop_obj:
                img, label = img.to(device), label.to(device)
                pred = ResNet(img)
                loss = loss_fn(pred, label)
                val_avg_loss.append(loss.item())
                pred = torch.argmax(pred, dim=1)
                accuracy = (pred == label).float().mean()
                loop_obj.set_postfix_str(f"Loss: {loss.item():0.3f},  Accuracy: {accuracy:0.3f}")
                val_avg_acc.append(accuracy)
            print(f"Validation Loss in this Epoch: {sum(val_avg_loss)/len(val_avg_loss)}")
            print(f"Validation Accuracy in this Epoch: {sum(val_avg_acc)/len(val_avg_acc)}")


def train_wb():
    run = wandb.init()
    config = wandb.config
    wandb.run.name = "lr_{}".format(config.lr)  
    
    torch.backends.cudnn.benchmark = True


    lr = config.lr

    ResNet.to(device)
    optim = Adam(ResNet.parameters(), lr = lr)
 
    for epoch in range(1, epochs+1):
        loop_obj = tqdm(train_loader)
        train_avg_loss, train_avg_acc = [], []
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
                train_avg_loss.append(loss.item())
                accuracy = (torch.argmax(pred, dim=1) == label).float().mean()
                train_avg_acc.append(accuracy)
                loop_obj.set_postfix_str(f"Loss: {loss.item():0.3f}, Accuracy: {accuracy:0.3f}")

        with torch.no_grad():
            loop_obj = tqdm(test_loader)
            val_avg_loss, val_avg_acc = [], []        
            ResNet.eval()
            for img, label in loop_obj:
                img, label = img.to(device), label.to(device)
                pred = ResNet(img)
                loss = loss_fn(pred, label)
                val_avg_loss.append(loss.item())
                pred = torch.argmax(pred, dim=1)
                accuracy = (pred == label).float().mean()
                loop_obj.set_postfix_str(f"Accuracy: {accuracy:0.3f}")
                val_avg_acc.append(accuracy)

        wandb.log({
                    "train_loss": sum(train_avg_loss)/len(train_avg_loss),
                    "train_accuracy": sum(train_avg_acc)/len(train_avg_acc),
                    "test_loss": sum(val_avg_loss)/len(val_avg_loss),
                    "test_accuracy": sum(val_avg_acc)/len(val_avg_acc)
        })


if __name__ == "__main__":
    if args.question:
        wandb.login(key="e99813e81e3838e6607d858a20693d589933495f")
        with open("./sweep2.yml", "r") as f:
            sweep_config = yaml.safe_load(f)
        
        sweep_id = wandb.sweep(sweep_config, project=args.wandb_project)    
        wandb.agent(sweep_id, function=train_wb)
        
    else :
        train()
