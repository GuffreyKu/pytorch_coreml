import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchvision import models
from torchmetrics import Accuracy


class Model(pl.LightningModule):
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
        self.log_dict({'train_loss': loss, 'train_acc': acc}, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        img, label = batch

        out = self(img)
        _, pred = out.max(1)

        loss = self.val_loss(out, label)
        acc = self.val_acc(pred, label)
        self.log_dict({'val_loss': loss, 'val_acc': acc})

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer)
        return {"optimizer": optimizer, "scheduler": lr_scheduler}
