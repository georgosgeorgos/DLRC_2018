import torch as th
from torch.utils import data
import pickle as pkl
import numpy as np

class Loader(data.Dataset):

    def __init__(self, split="train", transform=None):

        self.split = split

        if self.split == "train":
            with open("./data/train.pkl", "rb") as f:
                self.data = pkl.load(f)
                self.data = [self.data[i]["lidar"]["measure"] for i in range(len(self.data))]
                self.data = np.array(self.data, dtype=float)
                self.data = np.reshape(self.data, (-1, self.data.shape[2]))
                self.data /= 1000
                self.data[self.data > 2.0] = 2.0

        else:
            with open("./data/test.pkl", "rb") as f:
                self.data = pkl.load(f)[0]["data"]
                print(len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        lidar = self.data[index]
        if self.split == "train":
            lidar = th.from_numpy(lidar)
            lidar = lidar.float()
        else:
            lidar_list = [0 for i in range(9)]
            lidar_list[3] = lidar
            lidar_array = np.array(lidar_list)
            lidar_array = th.from_numpy(lidar_array)
            lidar_array = lidar_array.float()
        return lidar_array

if __name__ == '__main__':
    data = Loader("train")
    print(len(data))
    print(data[100])