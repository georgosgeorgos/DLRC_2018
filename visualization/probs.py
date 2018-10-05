import numpy as np
import torch as th
import glob
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data import DataLoader
import torch as th
from visualization.normalMLP import  NormalMLP

import os.path as osp
from torch.distributions.normal import Normal
import pickle as pkl 


pf = "./robot_sampling/data_0.pkl"
#pf = "../data/train_data_correct.pkl"

class Sampler:
    def __init__(self, n=100):
        with open(pf, "rb") as f:
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
            self.data_joint_v = np.array(self.data_joint_v, dtype=float)

        self.index=0
        self.n = 100

    def get_sample(self):
        if self.index == (self.data_lidar.shape[0] - self.n):
            self.index = 0
        sample_lidar   = self.data_lidar[(self.index):(self.index+self.n)]
        sample_joint   = self.data_joint[(self.index):(self.index+self.n)]
        sample_joint_v = self.data_joint_v[(self.index):(self.index+self.n)]
        self.index += self.n
        return (sample_lidar, sample_joint, sample_joint_v)

class Probs:
    def __init__(self, n=100, l=3):
        self.cuda = th.cuda.is_available()
        self.device = th.device("cuda" if self.cuda else "cpu")
        self.model = NormalMLP().to(self.device)
        self.model.load_state_dict(th.load("../experiments/normalMLP/ckpt/ckpt.pkl", map_location=self.device))
        self.model.eval()

        self.index=0
        self.n = n
        self.l = l
        self.sampler = Sampler(self.n)

    def get_data(self):
        y, x, _ = self.sampler.get_sample()

        data = {}
        x = th.from_numpy(x).to(self.device).float()
        y = th.from_numpy(y).to(self.device).float()

        mu, logvar = self.model(x)
        std  = th.exp(0.5 * logvar)
        N    = Normal(mu, std)
        prob = 1 - N.cdf(y)

        y    = y.cpu().data.numpy()[:,self.l]
        mu   = mu.cpu().data.numpy()[:,self.l]
        std  = std.cpu().data.numpy()[:,self.l]
        prob = prob.cpu().data.numpy()[:,self.l]


        self.old_prob = prob.copy()

        data =  {"input": y, "mu": mu, "std": std}
        return data

    def get_old_prob(self):
        return self.old_prob()

if __name__ == '__main__':
    p =Probs()
    d =p.get_data()
    for i in d.keys():
        print(i, d[i][:10,3])
        print()
