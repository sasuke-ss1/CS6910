from Model import Model
from dataset import NatureData
import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision import transforms
from torch.utils.data import random_split, DataLoader
from tqdm import tqdm
import yaml
import sys
import wandb
from argparse import ArgumentParser


parser = ArgumentParser()

parser.add_argument("--wandb_project", "-wp", default="test", type=str, help="Project name used to track experiments in Weights & Biases dashboard")
parser.add_argument("--wandb_entity", "-we", default="sasuke", type=str, help="Wandb Entity used to track experiments in the Weights & Biases dashboard.")
parser.add_argument("--batch_size", "-b", default=64, type=int, help="Batch size used to train neural network.")
parser.add_argument("--question", "-q", type=int, default=None, help="Set True to run wandb experiments")
parser.add_argument("--lr", "-lr", type=float, default=1e-3, help="Learning rate used to optimize model parameters")
parser.add_argument("--epochs", "-e", default=20, type=int, help="Number of epochs to train neural network.")
parser.add_argument("--num_filters", "-nf", default = 32, type=int, help="Base number of filters")
parser.add_argument("--filter_org", "-fo", type=str, default="const", help="Stratergy for depth of each layer's activation")
parser.add_argument("--activation", "-a", type=str, default="ReLU", help="Activation function after each layer")
parser.add_argument("--dropout", "-d", type=float, default=0.3, help="Dropout probability value for each layer")
parser.add_argument("--batch_norm", "-bn", type=bool, default=False, help="Set true to apply batch norm to every layer.")
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
            transforms.RandomCrop(224),
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
#train_data, val_data = random_split(data, [8000, 1999])
test_data = NatureData(parent_dir, False, val_transform)

train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, num_workers=2)
#val_loader = DataLoader(val_data, batch_size=batch_size)
val_loader = DataLoader(test_data, batch_size=batch_size)


def train():
    torch.backends.cudnn.benchmark = True
    print("Training begins\n")

    if args.filter_org != "const":
        filters = [3] + [int(args.num_filters*2**i) if args.filter_org == "double" else args.num_filters//2**(-i) for i in range(5)]
    else:
        filters = [3] + 5*[args.num_filters]

    loss_fn = nn.CrossEntropyLoss().to(device)
    model = Model(num_filters=filters, filter_size=3, pool_size=2, activation=args.activation, img_shape=224, dropout=args.dropout, batch_norm=args.batch_norm)
    model.to(device)
    optim = Adam(model.parameters(), lr = lr)
    
    for epoch in range(1, epochs+1):
        loop_obj = tqdm(train_loader)
        avg_loss, avg_acc = [], []
        model.train()
        for img, label in loop_obj:
            img, label = img.to(device), label.to(device)
            loop_obj.set_description(f"Epoch: {epoch}")
            optim.zero_grad()
            pred = model(img)
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
            val_avg_loss, val_avg_acc = [], []        
            model.eval()
            for img, label in loop_obj:
                img, label = img.to(device), label.to(device)
                pred = model(img)
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
    wandb.run.name = "lr_{}_nf_{}_dr_{}_fo_{}_ac_{}".format(config.lr, config.num_filters,\
                            config.dropout, config.filter_org, config.activation)  
    
    torch.backends.cudnn.benchmark = True

    if config.filter_org != "const":
        filters = [3] + [int(config.num_filters*2**i) if config.filter_org == "double" else config.num_filters//2**(-i) for i in range(5)]
    else:
        filters = [3] + 5*[config.num_filters]

    lr = config.lr
    loss_fn = nn.CrossEntropyLoss().to(device)
    Net = Model(num_filters=filters, filter_size=3, pool_size=2, activation=config.activation, img_shape=224, dropout=config.dropout, batch_norm=config.batch_norm)
    optim = Adam(Net.parameters(), lr = lr)
    
    for epoch in range(1, epochs+1):
        loop_obj = tqdm(train_loader)
        train_avg_loss, train_avg_acc = [], []
        Net.train()
        for img, label in loop_obj:
            img, label = img.to(device), label.to(device)
            loop_obj.set_description(f"Epoch: {epoch}")
            optim.zero_grad()
            pred = Net(img)
            loss = loss_fn(pred, label)
            loss.backward()
            optim.step()
            
            with torch.no_grad():
                train_avg_loss.append(loss.item())
                accuracy = (torch.argmax(pred, dim=1) == label).float().mean()
                train_avg_acc.append(accuracy)
                loop_obj.set_postfix_str(f"Loss: {loss.item():0.3f}, Accuracy: {accuracy:0.3f}")

        with torch.no_grad():
            loop_obj = tqdm(val_loader)
            val_avg_loss, val_avg_acc = [], []        
            Net.eval()
            for img, label in loop_obj:
                img, label = img.to(device), label.to(device)
                pred = Net(img)
                loss = loss_fn(pred, label)
                val_avg_loss.append(loss.item())
                pred = torch.argmax(pred, dim=1)
                accuracy = (pred == label).float().mean()
                loop_obj.set_postfix_str(f"Accuracy: {accuracy:0.3f}")
                val_avg_acc.append(accuracy)

        wandb.log({
                    "train_loss": sum(train_avg_loss)/len(train_avg_loss),
                    "train_accuracy": sum(train_avg_acc)/len(train_avg_acc),
                    "val_loss": sum(val_avg_loss)/len(val_avg_loss),
                    "val_accuracy": sum(train_avg_acc)/len(train_avg_acc)
        })




if __name__ == "__main__":

    if args.question:
        with open("./sweep.yml", "r") as f:
            sweep_config = yaml.safe_load(f)

        sweep_id = wandb.sweep(sweep_config, project=args.wandb_project)
        wandb.agent(sweep_id, function=train_wb, count=1)

        train_wb()

    else:
        train()