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

from utils.utils import move_to_cuda

def test(args):
    test_set    = PandaDataSetImg(root_dir=args.data_dir, split=args.split)
    test_loader = DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=False)
    model = Autoencoder()
    model = move_to_cuda(model)
    model.eval()

    saved_state_dict = th.load(args.ckpt_dir + args.ckpt_test)
    model.load_state_dict(saved_state_dict)

    MSE, T = 0, 0
    for iter, x in enumerate(test_loader):
        x_pred, _ = model(x)

        b, _, n, m = x.size()
        T   += n * m * b
        MSE += th.sum((x - x_pred)**2)

        print(MSE)

    print('RMSE: {:.4f}'.format(np.sqrt(MSE / T)))