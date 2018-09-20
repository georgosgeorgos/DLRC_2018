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

def main(args):

    ts = time.time()
    split = "train"

    def loss_fn(y_z, mu_phi, log_var_phi, mu_theta, log_var_theta, batch_size=24, n_sample=1, k=1):
        loglikelihood = 0
        
        std_theta = std(log_var_theta)
        std_phi   = std(log_var_phi)
        
        N = Normal(mu_theta, std_theta)

        y_expanded = th.cat([y_z, y_z, y_z], dim=1)
        pdf_y = th.exp(N.log_prob(y_expanded))
        pdf_y = reshape(pdf_y)

        # sample z to build empirical sample mean over z for the likelihood
        # we are using only one sample at time from the mixture ----> likelihood
        # is simply the normal
        for sample in range(n_sample):
            eps = V(th.randn(y_expanded.size()))
            z_y = eps * std_phi + mu_phi

            z_y = reshape(z_y)
            z_y = F.softmax(z_y, dim=2)
            loglikelihood += th.log(th.sum(pdf_y * z_y, dim=2))

        loglikelihood /= 10
        loglikelihood = th.sum(loglikelihood) / y_z.size()[0]*y_z.size()[1]
        # reduce mean over the batch size reduce sum over the lidars
        
        # reduce over KLD
        # explicit form when q(z|x) is normal and N(0,I)
        kld = 1/2 * torch.sum(log_var_phi.exp() + mu_phi.pow(2) - 1 - log_var_phi)
        # we want to maximize this guy
        elbo = loglikelihood - kld 

        return - elbo


    vae = VAE(
            encoder_layer_sizes=args.encoder_layer_sizes,
            latent_size=args.latent_size,
            batch_size=args.batch_size,
            conditional=args.conditional,
            num_labels= 7 if args.conditional else 0
            )

    optimizer = torch.optim.Adam(vae.parameters(), lr=args.learning_rate)

    tracker_global = defaultdict(torch.FloatTensor)

    dataset = Loader()
    data_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True)

    for epoch in range(args.epochs):
        print("Epoch: ", epoch)
        for iteration, y in enumerate(data_loader):

            # observable
            y = V(y)
            y = y / 1000
            #y = y.view(-1, 9)

            # conditioning
            #x = V(x)
            #x = x.view(-1, 7)

            if args.conditional:
                mu_phi, log_var_phi, mu_theta, log_var_theta = vae(y, x)
            else:
                mu_phi, log_var_phi, mu_theta, log_var_theta = vae(y)


            loss = loss_fn(y, mu_phi, log_var_phi, mu_theta, log_var_theta)

            if split == 'train':
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            tracker_global['loss'] = torch.cat( (tracker_global['loss'], torch.FloatTensor([loss.data/y.size(0)])), dim=0 )
            tracker_global['it'] = torch.cat((tracker_global['it'], torch.Tensor([epoch*len(data_loader)+iteration])))

            if iteration % args.print_every == 100 or iteration == len(data_loader)-1:
                print("Batch {:4d}/{}, Loss {:4.4f}".format(iteration, len(data_loader)-1, loss.item()))

                if args.conditional:
                    z_y = vae.inference(y)
                else:
                    z_y = vae.inference(y)

                print("inference:")
                print(z_y.data.numpy())

        #plt.plot(tracker_global["it"].data.numpy(), tracker_global["loss"].data.numpy())
        #plt.show()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--encoder_layer_sizes", type=list, default=[9, 256])
    #parser.add_argument("--decoder_layer_sizes", type=list, default=[256, 784])
    parser.add_argument("--latent_size", type=int, default=27)
    parser.add_argument("--print_every", type=int, default=100)
    parser.add_argument("--fig_root", type=str, default='figs')
    parser.add_argument("--conditional", action='store_true')

    args = parser.parse_args()

    main(args)
