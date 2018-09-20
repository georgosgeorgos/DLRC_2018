import torch as th
from torch.utils import data
import pickle as pkl
import numpy as np

class Loader(data.Dataset):

    def __init__(self, split="", transform=None):
        with open("../data/custom_trajectories.pkl", "rb") as f:
            self.data = pkl.load(f)[0]["data"]
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        lidar = self.data[index]

        lidar = th.from_numpy(np.array([lidar.astype(float) for _ in range(9)]))
        lidar = lidar.float()
        return lidar

if __name__ == '__main__':
    data = Loader()
    print(len(data))
    print(data[2])