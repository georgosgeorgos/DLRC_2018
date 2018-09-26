from torch.utils.data import Dataset
import pickle as pkl
import numpy as np
import os.path as osp
from PIL import Image
from libtiff import TIFF
import glob
import torch as th


class PandaDataSet(Dataset):
    def __init__(self, root_dir=None, train=None, test_split=0.2, transform=None):

        self.root_dir = root_dir
        self.train = train
        self.test_split = test_split
        self.transform = transform

        self.file_list = glob.glob(self.root_dir + "*")

    def __len__(self):
        """
        Return number of samples in the dataset, which is equivalent to flattening [num_demonstrations x num_timesteps]
        :return:
        """
        return len(self.file_list)

    def __getitem__(self, index):

        path_file = self.file_list[index] 
        tiff = TIFF.open(path_file, mode='r') 
        depth = tiff.read_image()
        depth = depth / 1000
        depth = depth.astype(float)
        tiff.close()
        depth = np.reshape(depth, (1, depth.shape[0], depth.shape[1]))
        depth = th.from_numpy(depth).float()

        return depth


if __name__ == '__main__':
    train_set = PandaDataSet("../../DEPTH/")
    print(train_set[0])
    print(train_set[0] / 1000)
    print(train_set[0].shape)
    print(train_set[0].dtype)
    print(type(train_set[0]))
    print(np.array(train_set[0]).dtype)