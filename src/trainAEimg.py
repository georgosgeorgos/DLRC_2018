import os
import time
import torch
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torch.distributions.normal import Normal
from collections import OrderedDict, defaultdict
import os.path as osp

import torch as th
import torch.nn.functional as F

from src.models.autoencoder.autoencoder import Autoencoder
from src.objectives.reconstruction import LossReconstruction
from src.loaders.load_panda_depth import PandaDataSetImg
from src.utils.utils import move_to_cuda, ckpt_utc, path_exists, tensor_to_variable, plot_eval

import datetime

lidar_input_size = 9  # number lidars obs var
joint_input_size = 7  # joint state   cond var
n_samples_y      = 10 # length timeseries
n_samples_z      = 10 # sample from selector
clusters         = 2  # clustering component (background/self | static/dynamic)
split            = "train"

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--learning_rate", type=float, default=0.0001)
parser.add_argument("--ckpt_dir", type=str, default="./ckpt/depth/")
parser.add_argument("--data_dir", type=str, default="../DEPTH/")
parser.add_argument("--split", type=str, default=split)

args = parser.parse_args()

path_results = osp.join('..', 'experiments', 'AEonDepthImg')
path_exists(path_results)
path_exists(osp.join(path_results, args.ckpt_dir))


def trainAE():

    split = args.split
    ckpt = ckpt_utc()

    train_set = PandaDataSetImg(root_dir=args.data_dir, split=args.split)
    train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True)

    loss_fn = LossReconstruction()
    loss_fn = move_to_cuda(loss_fn)

    model = Autoencoder()
    model = move_to_cuda(model)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    epoch_loss_history = []
    for epoch in range(args.epochs):
        loss_batch_history = []
        for iter, x in enumerate(train_loader):
            x = tensor_to_variable(x)
            if x.size(0) != args.batch_size:
                continue
            else:
                depth_pred, _ = model(x)
                loss = loss_fn(x, depth_pred)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                loss_batch_history.append(loss.item())

        epoch_loss = np.mean(loss_batch_history)
        epoch_loss_history.append(epoch_loss)

        print('train epoch: {} avg. loss: {:.4f}'.format(epoch, epoch_loss))

        plot_eval(np.arange(len(epoch_loss_history)), np.array(epoch_loss_history), save_to=osp.join(path_results, 'train_loss.png'), title = 'train loss')
        th.save(model.state_dict(), osp.join(path_results, args.ckpt_dir + ckpt))


if __name__ == '__main__':
    trainAE()