import numpy as np
import torch as th
import glob
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data import DataLoader
import torch as th
from load_panda import PandaDataSet
from normalMLP import NormalMLP
import os.path as osp
from torch.distributions.normal import Normal

class Sampler:
    def __init__(self, n=100, l=3):
        cuda = th.cuda.is_available()
        self.device = th.device("cuda" if cuda else "cpu")
        kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

        test_set = PandaDataSet(root_dir="../data/", filename='anomaly_detection_gt.pkl', train=False, test_split=0.0, transform=transforms.Compose([
            transforms.Lambda(lambda n: th.Tensor(n)),
            transforms.Lambda(lambda n: th.Tensor.clamp(n, 14, 2000)),
            transforms.Lambda(lambda n: n / 1000)
        ]))
        self.test_loader = DataLoader(test_set, batch_size=n, shuffle=False, **kwargs)

        self.model = NormalMLP().to(self.device)

        saved_state_dict = th.load("../experiments/normalMLP/ckpt/ckpt.pkl")[0]
        self.model.load_state_dict(saved_state_dict)
        self.model.eval()

        self.index=0

    def routine_index(self):
        if self.index >= (len(self.test_loader)-1):
            self.index = 0

    def get_sample(self):
        self.routine_index()

        data = {}
        (x, y, _) = test_loader[self.index]     
        x = x.to(self.device).float()
        y = y.to(self.device).float()

        mu, logvar = self.model(x)
        std = th.exp(0.5 * logvar)
        N = Normal(mu, std)
        anomaly_prob = N.log_prob(y)

        y            = y.cpu().data.numpy().tolist()
        mu           = mu.cpu().data.numpy().tolist()
        std          = std.cpu().data.numpy().tolist()
        pred         = y_sample.cpu().data.numpy().tolist()
        anomaly_prob = anomaly_prob.cpu().data.numpy().tolist()

        data =  {"input": y, "pred": pred, "mu": mu, "std": std, "anomaly_prob": anomaly_prob}
        return data

if __name__ == '__main__':
    s = Sampler()
    print(s.get_sample())