import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import random



class lvDataset(Dataset):
    def __init__(self, root='/mnt/liyinggang/tmp/LV/', train=True, transform_train=None
                 , transform_val=None):
        '''
        :param transform_train:
        :param transform_test:
        '''
        super(lvDataset, self).__init__()
        self.root = root
        self.train = train
        self.transform_val = transform_val
        self.transform_train = transform_train

        self._train_imgPath_label = []
        self._val_imgPath_label = []

        if self.train:
            with open(root + "Datasets/INSTRES1_train.txt") as f:
                for line in f.readlines():
                    newline = line.strip('  ').split('*')
                    self._train_imgPath_label.append((newline[0], newline[1]))
                random.shuffle(self._train_imgPath_label)

        else:
            with open(root + "Datasets/INSTRES1_val.txt") as test:
                for line in test.readlines():
                    newline = line.strip('  ').split('*')
                    self._val_imgPath_label.append((newline[0], newline[1]))
                random.shuffle(self._val_imgPath_label)



    def __getitem__(self, index):
        if self.train:
            imgPath, label = self._train_imgPath_label[index]
            label = int(label)

        else:
            imgPath, label = self._val_imgPath_label[index]
            label = int(label)

        img = Image.open(imgPath).convert('RGB')

        if self.train:
            img = self.transform_train(img)
        else:
            img = self.transform_val(img)
        return img, label

    def __len__(self):
        if self.train:
            return len(self._train_imgPath_label)
        else:
            return len(self._val_imgPath_label)



if __name__ == '__main__':
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    transform_train_global = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),  # 训练集才需要做这一步处理，获得更多的随机化数据
        transforms.ToTensor(),
        normalize])

    dataset = lvDataset(train=True, transform_train=transform_train_global)
    train_loader = DataLoader(dataset, batch_size=8, shuffle=True)
    for batch_id, (img_globals, labels) in enumerate(train_loader):
        img_globals, labels = \
            [x for x in (img_globals, labels)]
        if batch_id == 0:
            for img in img_globals:
                plt.figure('img')
                img = img.cpu().numpy()
                img = np.transpose(img, (1, 2, 0))
                print(labels[0].cpu().numpy())
                plt.imshow(img)
                plt.show()
        else:
            break
    print('over')