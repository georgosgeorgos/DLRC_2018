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

import datetime

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--batch_size", type=int, default=24)
parser.add_argument("--learning_rate", type=float, default=0.00001)
parser.add_argument("--encoder_layer_sizes", type=list, default=[9, 256])
parser.add_argument("--latent_size", type=int, default=27)
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


    def loss_fn(y_z, mu_phi, log_var_phi, mu_theta, log_var_theta, batch_size=24, n_sample=10):
        '''
        add annealing alpha (1-alpha)
        '''
        std_theta = std(log_var_theta)
        std_phi   = std(log_var_phi)

        N = Normal(mu_theta, std_theta)

        y_expanded = torch.cat([y_z, y_z, y_z], dim=1)
        #expand(y_z)
        pdf_y = th.exp(N.log_prob(y_expanded))
        pdf_y = reshape(pdf_y)
        # sample z to build empirical sample mean over z for the likelihood
        # we are using only one sample at time from the mixture ----> likelihood
        # is simply the normal
        loglikelihood = 0
        for sample in range(n_sample):
            eps = V(th.randn(y_expanded.size()))
            z_y = eps * std_phi + mu_phi

            z_y = reshape(z_y)
            z_y = F.softmax(z_y, dim=2)
            loglikelihood += th.log(th.sum(pdf_y * z_y, dim=2))

        loglikelihood /= n_sample
        loglikelihood = th.sum(loglikelihood, dim=1)
        loglikelihood = th.mean(loglikelihood) #/ y_z.size()[0]*y_z.size()[1]
        # reduce mean over the batch size reduce sum over the lidars
        
        # reduce over KLD
        # explicit form when q(z|x) is normal and N(0,I)
        # what about k? 9 or 27?
        k   = 1 #z_y.size()[2]
        kld = 0.5 * ((log_var_phi.exp() + mu_phi.pow(2) - log_var_phi) - k)
        kld = torch.sum(kld, dim=1)
        kld = torch.mean(kld)

        # we want to maximize this guy
        elbo = loglikelihood - kld
        # so we need to negate the elbo to minimize
        return -elbo, kld, loglikelihood


    model = VAE(
            encoder_layer_sizes=args.encoder_layer_sizes,
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
    #tracker_global = defaultdict(torch.FloatTensor)

    dataset = Loader(split=split)
    data_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True)
    print(len(data_loader))

    loss_list = []

    for epoch in range(args.epochs):
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

                loss, kld, ll = loss_fn(y, mu_phi, log_var_phi, mu_theta, log_var_theta, args.batch_size)

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

            #print("inference:")
            #print(z_y.data.numpy())
            #print("ll: ", -ll.data.numpy())
            #print("kld: ", kld.data.numpy())
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