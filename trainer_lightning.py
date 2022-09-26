import os
import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from py_utils.data_module import DataModule
from py_utils.module import Model

device = torch.device('mps' if torch.backends.mps.is_available() else "cpu")
epochs = 3

if __name__ == '__main__':

    data_paths = ['best_model', 'data']

    for path in data_paths:
        if not os.path.exists(path):
            os.mkdir(path)


    data_df = pd.read_csv('data/bird_data/birds.csv')

    data = DataModule(128, data_df)
    model = Model(400)

    checkpoint_callback = ModelCheckpoint(
        dirpath="best_model",
        save_top_k=1,
        filename="birds-{epoch:02d}-{val_loss:.2f}.ckpt",
        monitor="val_loss")

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator='auto',
        devices=1,
        logger=True,
        callbacks=[checkpoint_callback]
    )

    trainer.fit(model, data)
    # torch.save(model.state_dict(), "bird_resnet18.pkl")
    