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

from models.cvae_gmm.cvae_gmm import VAE
from objectives.nELBO_gmm import nELBO
from loaders.load_panda_timeseries import Loader

from utils.utils import move_to_cuda, ckpt_utc, path_exists, tensor_to_variable, plot_eval 

def test(args):

    split = args.split
    #10 samples y and 57% accuracy 
    
    model = VAE(
            encoder_layer_sizes=args.encoder_layer_sizes,
            decoder_layer_sizes=args.decoder_layer_sizes,
            latent_size=args.latent_size,
            batch_size=args.batch_size_test,
            conditional=args.conditional,
            num_labels=args.num_labels
            )
    
    saved_state_dict = th.load(args.ckpt_dir + args.ckpt)
    model.load_state_dict(saved_state_dict)
    
    dataset = Loader(split=args.split, samples=args.n_samples_y)
    data_loader = DataLoader(dataset=dataset, batch_size=args.batch_size_test, shuffle=False)

    prediction = []

    for epoch in range(args.epochs):
        for itr, y in enumerate(data_loader):
            # observable

            if args.conditional:
                z_y = model.inference(y, x)
            else:
                z_y = model.inference(y)

            z = z_y.data.numpy().squeeze()

            z = np.argmax(z, axis=1)
            prediction.append(z)

    prediction = np.array(prediction)
    np.savetxt("prediction.csv", prediction[:,3].astype(int), fmt='%i')
    print("Done!")