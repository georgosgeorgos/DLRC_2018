from torch.utils.data import Dataset, DataLoader
import pickle as pkl
import numpy as np
import os.path as osp
import torch as th
from src.utils.utils import path_exists
import utils.configs as cfg
from torchvision import transforms


class PandaDataSet(Dataset):
    def __init__(self, root_dir=None, train=None, test_split=0.2, transform=None):

        self.root_dir = root_dir
        self.train = train
        self.test_split = test_split
        self.transform = transform

        with open(osp.join(root_dir, 'train.pkl'), "rb") as f:
            self.data = pkl.load(f)
            self.num_demonstrations = len(self.data)
            self.num_timesteps = len(self.data[0]["lidar"]["measure"])
            self.num_samples = self.num_demonstrations*self.num_timesteps
            self.num_lidars = len(self.data[0]["lidar"]["measure"][0])
            self.num_joints = len(self.data[0]["state"]["j_pos"][0])

            # lidars [mm]
            self.Y = [self.data[i]["lidar"]["measure"] for i in range(self.num_demonstrations)]
            self.Y = np.array(self.Y, dtype=float).reshape((self.num_samples, self.num_lidars))
            # joint position [rad]
            self.X = [self.data[i]["state"]["j_pos"] for i in range(self.num_demonstrations)]
            self.X = np.array(self.X, dtype=float).reshape((self.num_samples, self.num_joints))
            # joint vel [rad/s]
            self.Z = [self.data[i]["state"]["j_vel"] for i in range(self.num_demonstrations)]
            self.Z = np.array(self.Z, dtype=float).reshape((self.num_samples, self.num_joints))

            if train is not None:
                if train:
                    train_idx = int(self.num_samples * (1 - self.test_split))
                    self.Y = self.Y[:train_idx]
                    self.X = self.X[:train_idx]
                    self.Z = self.Z[:train_idx]
                else:
                    test_idx = int(self.num_samples * (1 - self.test_split))
                    self.Y = self.Y[test_idx:]
                    self.X = self.X[test_idx:]
                    self.Z = self.Z[test_idx:]

    def __len__(self):
        """
        Return number of samples in the dataset, which is equivalent to flattening [num_demonstrations x num_timesteps]
        :return:
        """
        return self.Y.shape[0]

    def __getitem__(self, index):
        Y = self.Y[index]
        X = self.X[index]
        Z = self.Z[index]

        if self.transform is not None:
            Y = self.transform(Y)

        return X, Y, Z


if __name__ == '__main__':
    train_set = PandaDataSet(root_dir='../../data/data_toy/', train=True,
                             transform=transforms.Compose([
            transforms.Lambda(lambda n: th.Tensor(n)),
            transforms.Lambda(lambda n: th.Tensor.clamp(n, cfg.LIDAR_MIN_RANGE, cfg.LIDAR_MAX_RANGE)),
            transforms.Lambda(lambda n: n / 1000)
        ])
                             )
    train_loader = DataLoader(train_set, batch_size=256, shuffle=True, num_workers=1)

    print(len(train_loader.dataset))

    test_set = PandaDataSet(root_dir='../../data/data_toy/', train=False,
                             transform=transforms.Compose([
                                 transforms.Lambda(lambda n: th.Tensor(n)),
                                 transforms.Lambda(
                                     lambda n: th.Tensor.clamp(n, cfg.LIDAR_MIN_RANGE, cfg.LIDAR_MAX_RANGE)),
                                 transforms.Lambda(lambda n: n / 1000)
                             ])
                             )
    test_loader = DataLoader(test_set, batch_size=256, shuffle=True, num_workers=1)

    print(len(test_loader.dataset))
