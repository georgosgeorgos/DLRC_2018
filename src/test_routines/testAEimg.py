import os
import torch
import numpy as np
import pandas as pd
import seaborn as sns
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torch.distributions.normal import Normal
import os.path as osp

import torch as th

from models.autoencoder.autoencoder import Autoencoder
from objectives.reconstruction import LossReconstruction
from loaders.load_panda_depth import PandaDataSetImg

def test(args):
    test_set    = PandaDataSetImg(root_dir=args.data_dir, split=args.split)
    test_loader = DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=True)
    model = Autoencoder()
    model = move_to_cuda(model)
    model.eval()

    saved_state_dict = th.load(args.ckpt_dir, map_location='cpu')
    model.load_state_dict(saved_state_dict)

    T = 0
    MSE = 0
    for epoch in range(args.epochs):
        for iter, x in enumerate(test_loader):
            x_pred, _ = model(x)

            _, _, n, m = x.size()
            x      = x.data.numpy()
            x_pred = x_pred.data.numpy()
            
            T    += n * m
            MSE += ((x - x_pred)**2).sum()

        print('RMSE: {:.4f}'.format(np.sqrt(MSE / T)))