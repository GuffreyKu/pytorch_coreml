from torch.utils.data.dataset import Dataset
import torch
import cv2


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

    def __getitem__(self, index):
        img = cv2.imread(self.img_path_List[index])
        img = cv2.resize(img, (112, 112))/255.0
        img = img.astype(np.float32)
        imgTensor = torch.from_numpy(img.transpose((2, 0, 1)))
        return imgTensor, self.labelList[index]

    def __len__(self):
        return len(self.labelList)
