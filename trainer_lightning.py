import numpy as np
import pandas as pd
import cv2

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import random_split, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

import pytorch_lightning as pl
from torchmetrics import Accuracy

from py_dataset.image_dataset import ImageDataset
device = torch.device('mps' if torch.backends.mps.is_available() else "cpu")
epochs = 50


class DataModule(pl.LightningDataModule):
    def __init__(self, batch_size, datadf):
        super().__init__()
        self.batch_size = batch_size

        self.trainDataset = ImageDataset(datadf, 'train')
        self.validDataset = ImageDataset(datadf, 'valid')

    def train_dataloader(self):
        return DataLoader(self.trainDataset, batch_size=self.batch_size, drop_last=True, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.validDataset, batch_size=self.batch_size)

    # def test_dataloader(self):
    #     return DataLoader(self.cifar_test, batch_size=self.batch_size)


class SimpleModel(pl.LightningModule):
    def __init__(self, class_num):
        super().__init__()
        self.save_hyperparameters()
        self.model = models.resnet18(pretrained=True)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, class_num)
        self.train_loss = nn.CrossEntropyLoss()
        self.train_acc = Accuracy()
        self.val_loss = nn.CrossEntropyLoss()
        self.val_acc = Accuracy()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        img, label = batch

        out = self(img)
        _, pred = out.max(1)
        loss = self.train_loss(out, label)
        acc = self.train_acc(pred, label)
        self.log_dict({'train/loss': loss, 'train/acc': acc}, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        img, label = batch

        out = self(img)
        _, pred = out.max(1)

        loss = self.val_loss(out, label)
        acc = self.val_acc(pred, label)
        self.log_dict({'val/loss': loss, 'val/acc': acc})

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        # lr_scheduler_config = get_lr_scheduler_config(optimizer)
        return {"optimizer": optimizer}


class ce_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, output, target):
        return self.loss(output, target.long())


if __name__ == '__main__':
    data_df = pd.read_csv('data/bird_data/birds.csv')

    data = DataModule(64, data_df)
    model = SimpleModel(400)
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator='auto',
        devices=1,
        logger=True,
        deterministic=True,
    )
    trainer.fit(model, data)
