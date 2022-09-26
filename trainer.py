import numpy as np
import cv2
import os
import random
import time
import torch

import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

from basic_model.basic import CNNBlock, Fullyconnect
from py_dataset.image_dataset import ImageDataset

device = torch.device('mps' if torch.backends.mps.is_available() else "cpu")
epochs = 50


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
    # testDataset = ImageDataset(data_df, 'test')
    validDataset = ImageDataset(data_df, 'valid')

    dataLoaderTrain = torch.utils.data.DataLoader(trainDataset,
                                                  batch_size=64,
                                                  shuffle=True,
                                                  num_workers=0,
                                                  drop_last=True)

    dataLoaderValid = torch.utils.data.DataLoader(validDataset,
                                                  batch_size=64,
                                                  shuffle=True,
                                                  num_workers=0,
                                                  drop_last=True)

    # dataLoadeTest = torch.utils.data.DataLoader(testDataset,
    #                                             batch_size=64,
    #                                             shuffle=True,
    #                                             num_workers=0,
    #                                             drop_last=True)

    net = Architecture(numclass=399).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
    ceLoss = ce_loss().to(device)

    for i in range(epochs):
        val_loss = []
        for image, label in dataLoaderTrain:
            optimizer.zero_grad()
            image = image.to(device)
            label = label.to(device, dtype=torch.long)
            output = net(image)
            loss = ceLoss(output, label)
            loss.backward()
            optimizer.step()

        net.eval()
        with torch.no_grad():
            for image, label in dataLoaderValid:
                image = image.to(device)
                label = label.to(device, dtype=torch.long)
                output = net(image)
                loss = ceLoss(output, label)
                val_loss.append(loss.to('cpu'))
        print('val loss', np.mean(val_loss))

        # for image, label in dataLoadeTest:
        #     print(image.shape)
        #     print(label)
        #     break
