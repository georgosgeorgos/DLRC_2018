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

from models.autoencoder.autoencoder import Autoencoder
from objectives.reconstruction import LossReconstruction
from loaders.load_panda_depth import PandaDataSetImg
from utils.utils import move_to_cuda, ckpt_utc, path_exists, tensor_to_variable, plot_eval

from PIL import Image
import datetime

lidar_input_size = 9  # number lidars obs var
joint_input_size = 7  # joint state   cond var
n_samples_y      = 10 # length timeseries
n_samples_z      = 10 # sample from selector
clusters         = 2  # clustering component (background/self | static/dynamic)
split            = "test"

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--ckpt_dir", type=str, default="../experiments/ckpt/depth/")
parser.add_argument("--data_dir", type=str, default="../DEPTH/")
parser.add_argument("--split", type=str, default=split)

args = parser.parse_args()

path_results = osp.join('..', 'experiments', 'AEonDepthImg')
path_exists(path_results)
path_exists(osp.join(path_results, args.ckpt_dir))


def testAE():
    split = args.split
    ckpt  = "ckpt_depth.pth"

    train_set    = PandaDataSetImg(root_dir=args.data_dir, split=args.split)
    train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True)

    model = Autoencoder()
    model = move_to_cuda(model)
    model.eval()

    saved_state_dict = th.load(args.ckpt_dir + ckpt, map_location='cpu')
    model.load_state_dict(saved_state_dict)

    T = 0
    RMSE = 0
    i = 0

    for epoch in range(args.epochs):
        for iter, x in enumerate(train_loader):
            x_pred, _ = model(x)

            _, _, n, m = x.size()
            x      = x.data.numpy()
            x_pred = x_pred.data.numpy()
            
            T    += n * m
            RMSE += ((x - x_pred)**2).sum() 
            i += 1
            if i % 100 == 0:
                print(x)
                print(x_pred)
                print(RMSE / T)

        print('RMSE: {:.4f}'.format(RMSE / T))

if __name__ == '__main__':
    testAE()