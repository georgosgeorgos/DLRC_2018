import pickle as pkl
from random import shuffle
from torchvision import transforms
import numpy as np
import torch as th
from torch.utils import data
import utils.configs as cfg


class Loader(data.Dataset):
    def __init__(self, path="../data/", filename="train_data_correct.pkl", split="train", n_samples=10, pivot=0, 
                 is_transform=False, is_joint_v=True):
        self.split       = split
        self.path        = path
        self.n_samples   = n_samples
        self.index_lidar = []
        self.is_joint_v  = is_joint_v

        self.transform=transforms.Compose([
            transforms.Lambda(lambda n: th.Tensor(n)),
            transforms.Lambda(lambda n: th.Tensor.clamp(n, cfg.LIDAR_MIN_RANGE, cfg.LIDAR_MAX_RANGE)),
            transforms.Lambda(lambda n: n / 1000)])

        self.is_transform=is_transform

        if self.split not in ["test"]:
            with open(path + filename, "rb") as f:
                self.data = pkl.load(f)
        else:
            with open(path + "clustering_gt.pkl", "rb") as f:
                self.data = pkl.load(f)

        self.runs = len(self.data)
        self.data_lidar   = []
        self.data_joint   = []
        self.data_joint_v = []
        for i in range(self.runs):       
            self.data_lidar.extend(self.data[i]["lidar"]["measurements"])
            self.data_joint.extend(self.data[i]["state"]["j_pos"])
            self.data_joint_v.extend(self.data[i]["state"]["j_vel"])
        
        self.data_lidar = np.array(self.data_lidar, dtype=float)
        self.data_lidar /= 1000
        self.data_lidar[self.data_lidar > 2.0] = 2.0

        self.data_joint = np.array(self.data_joint, dtype=float)
        # joint vel [rad/s]
        self.data_joint_v = np.array(self.data_joint_v, dtype=float)

        self.n = self.data_lidar.shape[0]
        
    def generate_index(self):
        if self.split == "train":
            self.index_lidar = [i for i in range(0, int(0.8 * self.n))]
            shuffle(self.index_lidar)
        elif self.split == "val":
            self.index_lidar = [i for i in range(int(0.8 * self.n), self.n)]
        else:
            self.index_lidar = [i for i in range(self.n)]

        self.index_lidar = np.array(self.index_lidar)

    def __len__(self):
        return len(self.index_lidar)

    def routine(self, x_data, i):
        x = x_data[(i - self.n_samples):i]
        x = x.flatten()
        x = th.from_numpy(x).float()
        return x

    def __getitem__(self, index):
        if self.split not in ["test"]:
            i = self.index_lidar[index]
        else:
            i = index

        if i < self.n_samples:
                i += self.n_samples

        Y = self.data_lidar[(i - self.n_samples):i]
        Y = th.from_numpy(Y).float()

        X   = self.routine(self.data_joint, i)
        X_v = self.routine(self.data_joint_v, i)
        
        if self.is_joint_v:
            X = th.cat([X, X_v])

        if self.is_transform:
            lidar = self.transform(lidar)

        return Y, X

if __name__ == '__main__':

    data = Loader()
    print(len(data))
    #print(data[100])
    #data = Loader("tr")
    #print(data[100])