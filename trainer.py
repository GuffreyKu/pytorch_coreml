import numpy as np
import cv2
import os
import random
import time
import torch
from torch.utils.data.dataset import Dataset
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
device = torch.device(1 if torch.backends.mps.is_available() else "cpu")


class ImageDataset(Dataset):
    def __init__(self, data_df, key):
        super().__init__()
        self.img_path_List = []
        self.labelList = []

        data_df = data_df[data_df['data set'] == key]
        for idx in data_df.index:
            self.img_path_List.append(
                'data/bird_data/' + data_df.at[idx, 'filepaths'])
            self.labelList.append(int(data_df.at[idx, 'class index']))
        print(len(self.labelList))

    def __getitem__(self, index):
        img = cv2.imread(self.img_path_List[index])
        img = cv2.resize(img, (112, 112))/255.0
        img = img.astype(np.float32)
        imgTensor = torch.from_numpy(img.transpose((2, 0, 1)))
        return imgTensor, self.labelList[index]

    def __len__(self):
        return len(self.labelList)


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channel, k_size, pad=1, s=1, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channel,
                              kernel_size=k_size, padding=pad, stride=s, dilation=dilation)
        self.batchNorm = nn.BatchNorm2d(out_channel)
        self.actfunction = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.batchNorm(x)
        x = self.actfunction(x)
        return x


class Fullyconnect(nn.Module):
    def __init__(self, in_channels, out_channel):
        super().__init__()
        self.fc = nn.Linear(in_channels, out_channel)
        self.actfunction = nn.Sigmoid()

    def forward(self, x):
        x = self.fc(x)
        x = self.actfunction(x)
        return x


class Architecture(nn.Module):
    def __init__(self, numclass=10):
        super().__init__()
        self.convs = nn.Sequential(CNNBlock(in_channels=3, out_channel=32, k_size=3),
                                   nn.MaxPool2d(kernel_size=2, stride=2),
                                   CNNBlock(in_channels=32,
                                            out_channel=64, k_size=3),
                                   nn.MaxPool2d(kernel_size=2, stride=2),
                                   CNNBlock(in_channels=64,
                                            out_channel=128, k_size=3),
                                   nn.AdaptiveAvgPool2d((3, 3)),)
        self.fc = Fullyconnect(3*3*128, numclass)

    def forward(self, x):
        x = self.convs(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return F.softmax(x, dim=1)


class ce_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, output, target):
        return self.loss(output, target.long())


if __name__ == '__main__':
    data_df = pd.read_csv('data/bird_data/birds.csv')

    trainDataset = ImageDataset(data_df, 'train')
    testDataset = ImageDataset(data_df, 'test')
    validDataset = ImageDataset(data_df, 'valid')

    dataLoaderTrain = torch.utils.data.DataLoader(trainDataset,
                                                  batch_size=1,
                                                  shuffle=True,
                                                  num_workers=0,
                                                  drop_last=True)

    dataLoaderValid = torch.utils.data.DataLoader(validDataset,
                                                  batch_size=1,
                                                  shuffle=True,
                                                  num_workers=0,
                                                  drop_last=True)

    dataLoadeTest = torch.utils.data.DataLoader(testDataset,
                                                batch_size=1,
                                                shuffle=True,
                                                num_workers=0,
                                                drop_last=True)

    for image, label in dataLoaderTrain:
        print(image.shape)
        print(label)
        break

    for image, label in dataLoaderValid:
        print(image.shape)
        print(label)
        break

    for image, label in dataLoadeTest:
        print(image.shape)
        print(label)
        break
