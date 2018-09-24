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
from losses import nELBO
from lidarLoader import Loader

import datetime

lidar_input_size = 9  # number lidars obs var
joint_input_size = 7  # joint state   cond var
n_samples_y      = 10 # length timeseries
n_samples_z      = 10 # sample from selector
clusters         = 2  # clustering component (background/self | static/dynamic)

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--batch_size", type=int, default=24)
parser.add_argument("--learning_rate", type=float, default=0.0001)
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
    split = "train"

    s = datetime.datetime.utcnow()
    s = str(s).split(".")[0]
    s = s.split(" ")
    s = "_".join(s)
    ckpt = "ckpt_" + s + ".pth"
    print(ckpt)


    # #def loss_fn(y_z, mu_phi, log_var_phi, mu_theta, log_var_theta, batch_size=2, n_samples_z=10, n_samples_y=5):
    #     '''
    #     add annealing alpha (1-alpha)
    #     convert in a class
    #     '''
    #     std_theta = std(log_var_theta)
    #     std_phi   = std(log_var_phi)

    #     N = Normal(mu_theta, std_theta)

    #     # if input more than one sample
    #     #print(y_z.shape)
    #     #print(y_z[0,-9:])
    #     #y_z = th.mean(y_z.view(batch_size, input_size, n_samples_y), dim=2)
    #     y_z = y_z.view(-1, n_samples_y, input_size)
    #     y_z = y_z[:,-1,:]
    #     #print(y_z.shape)
    #     #print(y_z[0, -1, :])
    #     #print(y_z.size())

    #     y_expanded = torch.cat([y_z, y_z], dim=1)
    #     #expand(y_z)
    #     #print(y_expanded.size())
    #     pdf_y = th.exp(N.log_prob(y_expanded))
    #     pdf_y = reshape(pdf_y)
    #     #print(y_expanded)
    #     #print(pdf_y)
    #     # sample z to build empirical sample mean over z for the likelihood
    #     # we are using only one sample at time from the mixture ----> likelihood
    #     # is simply the normal
    #     loglikelihood = 0
    #     # for every sample compute the weighted mixture
    #     for sample in range(n_samples_z):
    #         eps = V(th.randn(y_expanded.size()))
    #         # we use z_y as a selector/weight (z_i is a three dimensional Gaussian
    #         # in this way we can also measure uncertainly)
    #         z_y = eps * std_phi + mu_phi

    #         z_y = reshape(z_y)
    #         z_y = F.softmax(z_y, dim=2)
    #         # log of mixture weighted with z
    #         loglikelihood += th.log(th.sum(pdf_y * z_y, dim=2))

    #     loglikelihood /= n_samples_z
    #     loglikelihood = th.sum(loglikelihood, dim=1)
    #     loglikelihood = th.mean(loglikelihood) #/ y_z.size()[0]*y_z.size()[1]
    #     # reduce mean over the batch size reduce sum over the lidars
        
    #     # reduce over KLD
    #     # explicit form when q(z|x) is normal and N(0,I)
    #     # what about k? 9 or 27?
    #     k   = 1 #z_y.size()[2]
    #     kld = 0.5 * ((log_var_phi.exp() + mu_phi.pow(2) - log_var_phi) - k)
    #     kld = torch.sum(kld, dim=1)
    #     kld = torch.mean(kld)

    #     # we want to maximize this guy
    #     elbo = loglikelihood - kld
    #     # so we need to negate the elbo to minimize
    #     return -elbo, kld, loglikelihood
    
    loss_fn = nELBO(args.batch_size, n_samples_z, n_samples_y)

    model = VAE(
            encoder_layer_sizes=args.encoder_layer_sizes,
            decoder_layer_sizes=args.decoder_layer_sizes,
            latent_size=args.latent_size,
            batch_size=args.batch_size,
            conditional=args.conditional,
            num_labels=args.num_labels
            )

    #model = nn.DataParallel(model)
    #model.cuda()

    # probably adam not the most appropriate algorithms
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    optimizer.zero_grad()

    dataset = Loader(split=split, samples=n_samples_y)
    # randomize an auxiliary index because we want to use sample of time-series (10 time steps)
    data_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=False)
    print(len(data_loader))

    loss_list = []

    for epoch in range(args.epochs):
        dataset.generate_index()
        print("Epoch: ", epoch)
        L = []
        for itr, y in enumerate(data_loader):
            # observable
            y = V(y)

            if y.size(0) != args.batch_size:
                continue
            else:
                if args.conditional:
                    mu_phi, log_var_phi, mu_theta, log_var_theta = model(y, x)
                else:
                    mu_phi, log_var_phi, mu_theta, log_var_theta = model(y)

                loss, kld, ll = loss_fn(y, mu_phi, log_var_phi, mu_theta, log_var_theta)

                if split == 'train':
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                # compute the loss averaging over epochs and dividing by batches
                L.append(loss.data.numpy())

        if True:
            if args.conditional:
                z_y = model.inference(y)
            else:
                z_y = model.inference(y)

            print("loss: ", np.mean(L))

        loss_list.append(np.mean(L) / (len(data_loader)))

    plt.plot(np.array(loss_list))
    plt.grid()
    plt.show()

    th.save(model.state_dict(), args.ckpt_dir + ckpt)
    print("done!")

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)