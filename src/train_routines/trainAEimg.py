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

import datetime

def trainAE(args):

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