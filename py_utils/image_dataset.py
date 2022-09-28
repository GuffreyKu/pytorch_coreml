from torch.utils.data.dataset import Dataset
import torch
import PIL
import numpy as np


class ImageDataset(Dataset):
    def __init__(self, data_df, key):
        super().__init__()
        self.img_path_List = []
        self.labelList = []
        labelname = {}
        data_df = data_df[data_df['data set'] == key]
        for idx in data_df.index:
            self.img_path_List.append(
                'data/bird_data/' + data_df.at[idx, 'filepaths'])
            self.labelList.append(int(data_df.at[idx, 'class index']))
            if key == 'train':
                labelname[int(data_df.at[idx, 'class index'])
                          ] = data_df.at[idx, 'labels'] + '\n'
        if key == 'train':
            path = 'data/bird_data/labelname.txt'
            f = open(path, 'w')
            f.writelines(list(labelname.values()))

    def __getitem__(self, index):
        img = PIL.Image.open(self.img_path_List[index])
        img = img.resize([112, 112], PIL.Image.ANTIALIAS)
        img = np.asarray(img).astype(np.float32)
        img = img / 255
        imgTensor = torch.from_numpy(img.transpose((2, 0, 1)))
        return imgTensor, self.labelList[index]

    def __len__(self):
        return len(self.labelList)
