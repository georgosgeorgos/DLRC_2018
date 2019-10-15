import pickle as pkl
import random
from torchvision import transforms
import numpy as np
import torch as th
from torch.utils import data
import glob
import torch as th
from libtiff import TIFF
from torch.utils.data import Dataset
import sys
import utils.configs as cfg
from numpy.random import random, randint, choice, seed, shuffle


class PandaDataSetTimeSeries(data.Dataset):
    def __init__(
        self,
        path="../../data/",
        path_images="../../data/TRAIN_DATA/DEPTH/",
        filename="train_data_correct.pkl",
        split="train",
        n_samples=10,
        pivot=0,
        is_transform=False,
        is_joint_v=False,
        is_label_y=False,
        is_multimodal=False,
    ):
        self.index_lidar = []
        self.split = split
        self.n_samples = n_samples
        self.is_joint_v = is_joint_v
        self.is_label_y = is_label_y
        self.is_transform = is_transform
        self.is_multimodal = is_multimodal

        if self.split not in ["test"]:
            with open(path + filename, "rb") as f:
                self.data = pkl.load(f)
            self.img_list = glob.glob(path_images + "TRAIN/" + "*")
        else:
            with open(path + "clustering_gt.pkl", "rb") as f:
                self.data = pkl.load(f)
            self.img_list = glob.glob(path_images + "TEST/" + "*")

        self.runs = len(self.data)
        self.data_lidar = []
        self.data_joint = []
        self.data_joint_v = []
        for i in range(self.runs):
            self.data_lidar.extend(self.data[i]["lidar"]["measurements"])
            self.data_joint.extend(self.data[i]["state"]["j_pos"])
            self.data_joint_v.extend(self.data[i]["state"]["j_vel"])

        self.data_lidar = np.array(self.data_lidar, dtype=float)
        self.data_lidar /= 1000
        self.data_lidar[self.data_lidar > 2.0] = 2.0

        self.data_joint = np.array(self.data_joint, dtype=float)
        self.data_joint_v = np.array(self.data_joint_v, dtype=float)

        self.img_list = np.array(self.img_list)

        self.n = self.data_lidar.shape[0]
        self.base = "/".join(self.img_list[0].split("/")[:-1])
        self.base = str(self.base)

        self.data_lidar_collision, self.labels_collision = self.generate_collision_labels()

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
        x = x_data[(i - self.n_samples) : i]
        # x = x.flatten()
        x = x.T.flatten()
        x = th.from_numpy(x).float()
        return x

    def read_images(self, index, is_depth=True):
        img_array = None
        for ix in range((index - self.n_samples), index):
            path_file = self.base + "/" + str(ix) + ".tiff"
            tiff = TIFF.open(str(path_file), mode="r")
            img = tiff.read_image()
            if is_depth:
                img = img / 1000
                img[img > 2.0] = 2.0
            img = img.astype(float)
            img = np.reshape(img, (1, img.shape[0], img.shape[1]))
            tiff.close()
            if img_array is None:
                img_array = img.copy()
            else:
                img_array = np.vstack([img_array, img])
        return img_array

    # not the best way
    def generate_collision_labels_(self, y, p=0.3, t=0.040):
        cols = [i for i in range(y.shape[1])]
        numpy.random.seed()
        if np.random.random() < p:
            index = np.random.choice(cols, np.random.randint(1, y.shape[1] // 2), replace=False)
            y[:, index] = (
                np.random.random((y.shape[0], len(index))) * 2
            )  # (1 + 0.1 * np.random.randn(index.shape[0])) * t
        else:
            index = []

        # labels = np.zeros((y.shape[1], 2), dtype=int)
        labels = np.zeros(y.shape[1], dtype=int)
        labels[index] = 1
        # for c in cols:
        #    if c in index:
        #        labels[c,1] = 1
        #    else:
        #        labels[c,0] = 1
        return y, labels

    # one time computation
    def generate_collision_labels(self, p=0.3, t=0.040):
        y = self.data_lidar.copy()
        cols = [i for i in range(y.shape[1])]
        seed(1)
        labels = np.zeros(y.shape, dtype=int)
        s = self.n_samples
        for row in range(0, y.shape[0], self.n_samples):
            if random() < p:
                index = choice(cols, randint(1, y.shape[1] // 2), replace=False)
                y[(row) : ((row + s)), index] = (
                    np.random.random((s, len(index))) * 2
                )  # (1 + 0.1 * np.random.randn(index.shape[0])) * t
                labels[(row) : ((row + s)), index] = 1
        return y, labels

    def __getitem__(self, index):

        if self.split not in ["test"]:
            i = self.index_lidar[index]
        else:
            i = index

        if i < self.n_samples:
            i += self.n_samples

        if self.is_multimodal:
            depth = self.read_images(i)

        Y = self.data_lidar[(i - self.n_samples) : i]
        if self.is_label_y:
            Y = self.data_lidar_collision[(i - self.n_samples) : i]
            Y = Y.T.flatten()
            labels = self.labels_collision[(i - self.n_samples) : i]
            labels = th.from_numpy(labels).long()
            labels = labels[-1]

        Y = th.from_numpy(Y).float()
        X = self.routine(self.data_joint, i)
        X_v = self.routine(self.data_joint_v, i)

        if self.is_joint_v:
            X = th.cat([X, X_v])

        # if self.is_transform:
        if self.is_multimodal:
            if self.is_label_y:
                return Y, X, depth, labels
            else:
                return Y, X, depth
        else:
            if self.is_label_y:
                return Y, X, labels
            else:
                return Y, X
        return Y, X, depth


if __name__ == "__main__":

    data = PandaDataSetTimeSeries()
    print(data.read_images(29).shape)
    # print(data[100])
    # data = Loader("tr")
    # print(data[100])
