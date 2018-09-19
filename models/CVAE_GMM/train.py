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

from utils import *
from models import VAE

def main(args):

    ts = time.time()

    datasets = OrderedDict()
    datasets['train'] = MNIST(root='data', train=True, transform=transforms.ToTensor(), download=True)

    def loss_fn(y_z, mu_phi, log_var_phi, mu_theta, log_var_theta, batch_size=24):
        from torch.distributions.normal import Normal

        loglikelihood = 0
        
        std_theta = std(log_var_theta)
        std_phi   = std(log_var_phi)
        
        N = Normal(mu_theta, std_theta)

        y_expanded = th.cat([y_z, y_z, y_z], dim=1)
        pdf_y = th.exp(N.log_prob(y_expanded))
        pdf_y = reshape(pdf_y)

        # sample z to build likelihood over z
        for sample in range(10):
            eps = V(th.randn(y_expanded.size()))
            z_y = eps * std_phi + mu_phi

            z_y = reshape(z_y)
            z_y = F.softmax(z_y, dim=2)
            loglikelihood += th.log(th.sum(pdf_y * z_y, dim=2))

        loglikelihood = th.sum(loglikelihood) / y_z.size()[0]*y_z.size()[1]
        # reduce mean over the batch size reduce sum over the lidars
        
        # reduce over KLD
        KLD = 0.5 * torch.sum(1 + log_var_phi - mu_phi.pow(2) - log_var_phi.exp())

        return - (loglikelihood + KLD)


    vae = VAE(
            encoder_layer_sizes=args.encoder_layer_sizes,
            latent_size=args.latent_size,
            batch_size=args.batch_size,
            conditional=args.conditional,
            num_labels= 7 if args.conditional else 0
            )

    optimizer = torch.optim.Adam(vae.parameters(), lr=args.learning_rate)

    tracker_global = defaultdict(torch.FloatTensor)

    for epoch in range(args.epochs):
        for split, dataset in datasets.items():
            data_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=split=='train')

            for iteration, (y, x) in enumerate(data_loader):

                # observable
                y = V(y)
                y = y.view(-1, 9)

                # conditioning
                x = V(x)
                x = x.view(-1, 7)

                if args.conditional:
                    y, mu_phi, log_var_phi, mu_theta, log_var_theta = vae(y, x)
                else:
                    y, mu_phi, log_var_phi, mu_theta, log_var_theta = vae(y)


                loss = loss_fn(y, mu_phi, log_var_phi, mu_theta, log_var_theta)

                if split == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                tracker_global['loss'] = torch.cat( (tracker_global['loss'], torch.FloatTensor([loss.data/x.size(0)])), dim=0 )
                tracker_global['it'] = torch.cat((tracker_global['it'], torch.Tensor([epoch*len(data_loader)+iteration])))

                if iteration % args.print_every == 0 or iteration == len(data_loader)-1:
                    print("Batch {:4d}/{}, Loss {:4.4f}".format(iteration, len(data_loader)-1, loss.item()))

                    if args.conditional:
                        c=to_var(torch.arange(0,10).long().view(-1,1))
                        x = vae.inference(n=c.size(0), c=c)
                    else:
                        x = vae.inference(n=10)

                    plt.figure()
                    plt.figure(figsize=(5,10))
                    for p in range(10):
                        plt.subplot(5,2,p+1)
                        if args.conditional:
                            plt.text(0,0,"c=%i"%c.data[p][0], color='black', backgroundcolor='white', fontsize=8)
                        plt.imshow(x[p].view(28,28).data.numpy())
                        plt.axis('off')

                    if not os.path.exists(os.path.join(args.fig_root, str(ts))):
                        if not(os.path.exists(os.path.join(args.fig_root))):
                            os.mkdir(os.path.join(args.fig_root))
                        os.mkdir(os.path.join(args.fig_root, str(ts)))

                    plt.savefig(os.path.join(args.fig_root, str(ts), "E{}I{}.png".format(epoch, iteration)), dpi=300)
                    plt.clf()
                    plt.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--encoder_layer_sizes", type=list, default=[784, 256])
    parser.add_argument("--decoder_layer_sizes", type=list, default=[256, 784])
    parser.add_argument("--latent_size", type=int, default=27)
    parser.add_argument("--print_every", type=int, default=100)
    parser.add_argument("--fig_root", type=str, default='figs')
    parser.add_argument("--conditional", action='store_true')

    args = parser.parse_args()

    main(args)
