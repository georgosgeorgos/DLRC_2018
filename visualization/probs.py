import numpy as np
import torch as th
import glob
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data import DataLoader
import torch as th
from sampler import Sampler
from normalMLP import NormalMLP
import os.path as osp
from torch.distributions.normal import Normal

class Probs:
    def __init__(self, n=100, l=3):
        self.cuda = th.cuda.is_available()
        self.device = th.device("cuda" if self.cuda else "cpu")
        self.model = NormalMLP().to(self.device)
        self.model.load_state_dict(th.load("../experiments/normalMLP/ckpt/ckpt.pkl"))
        self.model.eval()

        self.index=0
        self.n = n
        self.l = l
        self.sampler = Sampler(self.n)

    def get_data(self):
        y, x, _ = self.sampler.get_sample()
        print(x.shape, y.shape)

        data = {}
        x = th.from_numpy(x).to(self.device).float()
        y = th.from_numpy(y).to(self.device).float()

        mu, logvar = self.model(x)
        std = th.exp(0.5 * logvar)
        N = Normal(mu, std)
        prob = N.log_prob(y)

        y            = y.cpu().data.numpy()
        mu           = mu.cpu().data.numpy()
        std          = std.cpu().data.numpy()
        prob         = prob.cpu().data.numpy()

        data =  {"input": y, "mu": mu, "std": std, "prob": prob}
        return data

if __name__ == '__main__':
    p =Probs()
    d =p.get_data()
    for i in d.keys():
        print(i, d[i][:10,3])
        print()