from torch.utils.data import Dataset, DataLoader
import os
from glob import glob
from PIL import Image
import torch
from torchvision import transforms
import sys

class NatureData(Dataset):
    def __init__(self, dir, train=True, transforms=None):
        super().__init__()
        self.train_dir = os.path.join(dir, "train")
        self.test_dir = os.path.join(dir ,"val")
        self.transforms = transforms
        self.classes = list(map(lambda x: x.split("/")[-1], glob(self.train_dir + "/*")))
        self.idx = [i for i in range(len(self.classes))]
        self.train = train
        self.cltoidx = dict(zip(self.classes, self.idx))

        self.train_img_path, self.test_img_path = [], []
        for cl in self.classes:
            self.train_img_path += glob(os.path.join(self.train_dir, cl) + "/*")
            self.test_img_path += glob(os.path.join(self.test_dir, cl) + "/*")

    def __len__(self):
        if self.train:
            return len(self.train_img_path)

        return len(self.test_img_path)

    def __getitem__(self, idx):
        if self.train:
            path = self.train_img_path[idx]
        else:
            path = self.test_img_path[idx]
        index = self.cltoidx[path.split("/")[-2]]
        img = Image.open(path).convert('RGB').resize((150, 150))
        label = index

        if self.transforms:
            img = self.transforms(img)

            return img, label

if __name__ == "__main__":
    transform = transforms.Compose([transforms.ToTensor(), transforms.RandomCrop((224, 224))])
    data = NatureData("./nature_12K", train=True, transforms = transform)
    loader = DataLoader(data, shuffle=True, batch_size=16)

    for img, label in loader:
        print(img.shape)
        print(label)
        sys.exit()