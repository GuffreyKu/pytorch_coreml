import PIL
import coremltools as ct
import numpy as np
import time
import pandas as pd
import torch
import cv2
from py_utils.module import Model

# device = torch.device('mps' if torch.backends.mps.is_available() else "cpu")
device = torch.device("cpu")


X = torch.rand(1, 3, 112, 112)
model = torch.jit.trace(torch.load('best_model/model_trace.pt'), X)
model = model.to(device)
bird_data = pd.read_csv('data/bird_data/birds.csv')
bird_data = bird_data[bird_data['data set'] == 'test']

if __name__ == '__main__':
    all_time = []
    for idx in bird_data.index:
        img_path = "data/bird_data/" + bird_data.at[idx, 'filepaths']
        img = PIL.Image.open(img_path)
        img = img.resize([112, 112], PIL.Image.ANTIALIAS)
        img = np.asarray(img).astype(np.float32)
        img = np.expand_dims(img, axis=0)
        imgTensor = torch.from_numpy(img.transpose((0, 3, 1, 2)))

        imgTensor = imgTensor.to(device)
        start = time.time()
        output = model(imgTensor)
        end = time.time()
        all_time.append(end - start)

        output = torch.argmax(output)

    print(np.sum(all_time))
    print(np.mean(all_time))
