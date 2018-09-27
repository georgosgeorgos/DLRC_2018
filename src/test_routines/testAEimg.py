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

def test(args):
    split = args.split
    

    train_set    = PandaDataSetImg(root_dir=args.data_dir, split=args.split)
    train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True)
    model = Autoencoder()
    #model = move_to_cuda(model)
    model.eval()

    saved_state_dict = th.load(args.ckpt_dir + args.ckpt, map_location='cpu')
    model.load_state_dict(saved_state_dict)

    T = 0
    MSE = 0
    i = 0

    for epoch in range(args.epochs):
        for iter, x in enumerate(train_loader):
            x_pred, _ = model(x)

            _, _, n, m = x.size()
            x      = x.data.numpy()
            x_pred = x_pred.data.numpy()
            
            T    += n * m
            MSE += ((x - x_pred)**2).sum()
            i += 1
            if i % 100 == 0:
                print(x)
                print(x_pred)
                print(MSE / T)

        print('RMSE: {:.4f}'.format(np.sqrt(MSE / T)))