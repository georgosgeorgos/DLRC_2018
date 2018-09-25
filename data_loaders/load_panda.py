from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch
import pickle as pkl
import numpy as np
import os.path as osp
from utils.utils import path_exists
import configs as cfg


class PandaDataSet(Dataset):
    def __init__(self, root_dir='data_toy', train=None, transform=None):

        self.root_dir = root_dir
        self.train = train
        self.transform = transform

        path_exists(root_dir)

        with open(osp.join(root_dir, 'train.pkl'), "rb") as f:
            self.data = pkl.load(f)
            self.num_demonstrations = len(self.data)
            self.num_timesteps = len(self.data[0]["lidar"]["measure"])
            self.num_samples = self.num_demonstrations*self.num_timesteps

            self.num_lidars = len(self.data[0]["lidar"]["measure"][0])
            self.num_joints = len(self.data[0]["state"]["j_pos"][0])

            self.Y = [self.data[i]["lidar"]["measure"] for i in range(self.num_demonstrations)]
            self.Y = np.array(self.Y, dtype=float).reshape((self.num_samples, self.num_lidars))

            self.X = [self.data[i]["state"]["j_pos"] for i in range(self.num_demonstrations)]
            self.X = np.array(self.X, dtype=float).reshape((self.num_samples, self.num_joints))

    def __len__(self):
        """
        Return number of samples in the dataset, which is equivalent to flattening [num_demonstrations x num_timesteps]
        :return:
        """
        return self.num_samples

    def __getitem__(self, index):
        Y = self.Y[index]
        X = self.X[index]

        if self.transform is not None:
            Y = self.transform(Y)

        return X, Y


# if __name__ == '__main__':
#     ds = PandaDataSet(
#         transform=transforms.Compose([
#             transforms.Lambda(lambda n: torch.Tensor(n)),
#             transforms.Lambda(lambda n: torch.Tensor.clamp(n, cfg.LIDAR_MIN_RANGE, cfg.LIDAR_MAX_RANGE)),
#             transforms.Lambda(lambda n: n / 1000)
#         ])
#     )
#     dl = DataLoader(ds, batch_size=1024, shuffle=True, num_workers=1)
#     for i, (x, y) in enumerate(dl):
#         print(torch.Tensor.min(y).item(), torch.Tensor.max(y).item())