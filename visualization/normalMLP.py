import os.path as osp

import numpy as np
import torch as th
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms

############################################################
### INITIALIZATION
############################################################

n_joints = 7
n_lidars = 9
n_hidden = 128
verbose = False
lr = 1e-3
every_nth_batch = 100
every_nth_epoch = 20
trbs = 512
tebs = 256
epochs = 1000
test_every_nth = 1
th.manual_seed(42)
cuda = th.cuda.is_available()
device = th.device("cuda" if cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
rootdir = '../../../'


############################################################
### MODEL
############################################################


class NormalMLP(nn.Module):

    def __init__(self):
        super(NormalMLP, self).__init__()

        self.fc1 = nn.Linear(n_joints, n_hidden)  # num_joints x num_hidden
        self.fc2 = nn.Linear(n_hidden, n_hidden)

        self.mu_layer = nn.Linear(n_hidden, n_lidars)  # num_hidden x num_channels
        self.logvar_layer = nn.Linear(n_hidden, n_lidars)

    def forward(self, *input):

        h1 = th.tanh(self.fc1(input[0]))
        h2 = th.tanh(self.fc2(h1))

        return self.mu_layer(h2), self.logvar_layer(h2)


