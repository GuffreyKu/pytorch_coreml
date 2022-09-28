import PIL
import coremltools as ct
import numpy as np
import time
import pandas as pd
import torch

model = ct.models.MLModel('best_model/bird.mlmodel')

bird_data = pd.read_csv('data/bird_data/birds.csv')
bird_data = bird_data[bird_data['data set'] == 'test']
if __name__ == '__main__':

    all_time = []
    for idx in bird_data.index:
        img_path = "data/bird_data/" + bird_data.at[idx, 'filepaths']
        img = PIL.Image.open(img_path)
        img = img.resize([112, 112], PIL.Image.ANTIALIAS)

        start = time.time()
        coreml_out_dict = model.predict({"input_1": img})
        end = time.time()
        all_time.append(end - start)
        print("top class label: ", coreml_out_dict["classLabel"])

    print(np.sum(all_time))
    print(np.mean(all_time))
    # print(np.argmax(coreml_out_dict['var_324']))
