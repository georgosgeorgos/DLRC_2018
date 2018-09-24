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

from utils import *
from model import VAE
from lidarLoader import Loader

lidar_input_size = 9  # number lidars obs var
joint_input_size = 7  # joint state   cond var
n_samples_y      = 10 # length timeseries
n_samples_z      = 10 # sample from selector
clusters         = 2  # clustering component (background/self | static/dynamic)

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--encoder_layer_sizes", type=list, default=[(lidar_input_size*n_samples_y), 256, 256])
parser.add_argument("--decoder_layer_sizes", type=list, default=[(joint_input_size*n_samples_y), 256, 256])
parser.add_argument("--latent_size", type=int, default=lidar_input_size*clusters)
parser.add_argument("--print_every", type=int, default=1000)
parser.add_argument("--fig_root", type=str, default='figs')
parser.add_argument("--conditional", action='store_true')
parser.add_argument("--num_labels", type=int, default=0)
parser.add_argument("--ckpt_dir", type=str, default="./ckpt/")

def main(args):

    ts = time.time()
    split = "test"
    #10 samples y and 57% accuracy 
    ckpt = "ckpt_2018-09-24_10:00:14.pth"
    

    model = VAE(
            encoder_layer_sizes=args.encoder_layer_sizes,
            decoder_layer_sizes=args.decoder_layer_sizes,
            latent_size=args.latent_size,
            batch_size=args.batch_size,
            conditional=args.conditional,
            num_labels=args.num_labels
            )
    #model = nn.DataParallel(model)
    saved_state_dict = th.load(args.ckpt_dir + ckpt)
    model.load_state_dict(saved_state_dict)

    dataset = Loader(split=split, samples=n_samples_y)
    data_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=False)

    prediction = []

    for epoch in range(args.epochs):
        for itr, y in enumerate(data_loader):
            print(y.size())
            print()
            # observable
            y = V(y)

            if args.conditional:
                z_y = model.inference(y)
            else:
                z_y = model.inference(y)

            z = z_y.data.numpy().squeeze()
            z = np.argmax(z, axis=1)
            prediction.append(z)

    prediction = np.array(prediction)
    np.savetxt("prediction.csv", prediction[:,3].astype(int), fmt='%i')
if __name__ == '__main__':
    args = parser.parse_args()
    main(args)