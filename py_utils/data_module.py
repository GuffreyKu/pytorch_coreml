import pandas as pd

import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl

from py_utils.image_dataset import ImageDataset


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
