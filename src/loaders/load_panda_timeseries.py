import pickle as pkl
from random import shuffle

import numpy as np
import torch as th
from torch.utils import data


class Loader(data.Dataset):
    def __init__(self, path="../data_mockup/", split="train", transform=None, samples=10, pivot=0):
        self.split   = split
        self.path    = path
        self.samples = samples
        self.index_lidar = []
        if self.split == "train" or self.split == "val":
            with open(path + "train.pkl", "rb") as f:
                self.data = pkl.load(f)

                self.data_lidar = [self.data[i]["lidar"]["measure"] for i in range(len(self.data))]
                self.data_lidar = np.array(self.data_lidar, dtype=float)
                self.data_lidar = np.reshape(self.data_lidar, (-1, self.data_lidar.shape[2]))
                self.data_lidar /= 1000
                self.data_lidar[self.data_lidar > 2.0] = 2.0

                self.data_joint = [self.data[i]["state"]["j_pos"] for i in range(len(self.data))]
                self.data_joint = np.array(self.data_joint, dtype=float)
                self.data_joint = np.reshape(self.data_joint, (-1, self.data_joint.shape[2]))
        else:
            with open(path + "test.pkl", "rb") as f:
                self.data_lidar = pkl.load(f)[0]["data"]
                self.data_lidar = np.array([float(i/1000) for i in self.data_lidar])

    def generate_index(self):
        n = self.data_lidar.shape[0]

        if self.split == "train":
            self.index_lidar = [i for i in range(0, int(2/3 * n))]
            shuffle(self.index_lidar)
        elif self.split == "val":
            self.index_lidar = [i for i in range(int(2/3 * n), n)]

        self.index_lidar = np.array(self.index_lidar)

    def __len__(self):
        return len(self.index_lidar)

    def __getitem__(self, index):
        if self.split == "train" or self.split == "val":
            ix = self.index_lidar[index]
        else:
            ix = index

        if ix < self.samples:
                ix += self.samples
        lidar = self.data_lidar[(ix - self.samples):ix]
        if self.split == "train" or self.split == "val":
            joint = self.data_joint[(ix - self.samples):ix]
            lidar = lidar.flatten()
            joint = joint.flatten()
            
            lidar = th.from_numpy(lidar)
            joint = th.from_numpy(joint)
            lidar = lidar.float()
            joint = joint.float()
        else:
            # everything background
            lidar_array = np.ones((self.samples, 9)) * 2.0
            # lidar 3
            k = 3
            lidar_array[:, k] = lidar
            lidar_array = lidar_array.flatten()
            #lidar_array = np.array(lidar_array)
            lidar_array = th.from_numpy(lidar_array)
            lidar = lidar_array.float()
            return lidar
        return lidar, joint

if __name__ == '__main__':

    data = Loader()
    print(len(data))
    #print(data[100])
    #data = Loader("tr")
    #print(data[100])