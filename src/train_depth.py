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

import torch as th
import torch.nn.functional as F

from models.autoencoder.autoencoder import Autoencoder as Model
from objectives.reconstruction import LossReconstruction as Loss
from loaders.load_panda_depth import PandaDataSet as Loader
from utils.utils import * 

import datetime

lidar_input_size = 9  # number lidars obs var
joint_input_size = 7  # joint state   cond var
n_samples_y      = 10 # length timeseries
n_samples_z      = 10 # sample from selector
clusters         = 2  # clustering component (background/self | static/dynamic)
split            = "train"

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--learning_rate", type=float, default=0.0001)
parser.add_argument("--encoder_layer_sizes", type=list, default=[(lidar_input_size*n_samples_y), 256, 256])
parser.add_argument("--decoder_layer_sizes", type=list, default=[(joint_input_size*n_samples_y), 256, 256])
parser.add_argument("--latent_size", type=int, default=lidar_input_size*clusters)
parser.add_argument("--fig_root", type=str, default='figs')
parser.add_argument("--conditional", action='store_true')
parser.add_argument("--num_labels", type=int, default=0)
parser.add_argument("--ckpt_dir", type=str, default="./ckpt/depth/")
parser.add_argument("--data_dir", type=str, default="../DEPTH/")
parser.add_argument("--split", type=str, default=split)


def main(args):
    split = args.split

    ckpt = ckpt_utc()
    print(ckpt)
    
    loss_fn = Loss()

    model   = Model()
    model   = check_cuda(model)
    loss_fn = check_cuda(loss_fn)

    model.train()

    # probably adam not the most appropriate algorithms
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    optimizer.zero_grad()

    dataset = Loader(root_dir=args.data_dir)
    data_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True)

    loss_list = []
    for epoch in range(args.epochs):
        print("Epoch: ", epoch)
        L = []
        for itr, depth in enumerate(data_loader):
            # observable
            depth = V(depth)
            if depth.size(0) != args.batch_size:
                continue
            else:
                depth_pred, _ = model(depth)
                loss = loss_fn(depth, depth_pred)

                if split == 'train':
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                # compute the loss averaging over epochs and dividing by batches
                L.append(loss.cpu().data.numpy())
                
        print("loss:", loss)
        
        loss_list.append(np.mean(L) / (len(data_loader)))

    plt.plot(np.array(loss_list))
    plt.grid()
    plt.show()

    path_exists(args.ckpt_dir)
    th.save(model.state_dict(), args.ckpt_dir + ckpt)
    print("done!")

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)