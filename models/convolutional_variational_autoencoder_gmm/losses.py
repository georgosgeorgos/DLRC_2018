import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

from utils import *

class nELBO(nn.Module):
    def __init__(self, batch_size=2, n_samples_z=10, n_samples_y=10):
        super(nELBO, self).__init__()
        '''
        add annealing alpha (1-alpha)
        convert in a class
        '''
        self.batch_size  = batch_size
        self.n_samples_z = n_samples_z
        self.n_samples_y = n_samples_y
        
        self.input_size = 9

    def forward(self, y_z, mu_phi, log_var_phi, mu_theta, log_var_theta):
        std_theta = std(log_var_theta)
        std_phi   = std(log_var_phi)
        N = Normal(mu_theta, std_theta)

        # if input more than one sample
        #print(y_z.shape)
        #print(y_z[0,-9:])
        #y_z = th.mean(y_z.view(batch_size, input_size, n_samples_y), dim=2)
        y_z = y_z.view(-1, self.n_samples_y, self.input_size)
        y_z = y_z[:,-1,:]
        #print(y_z.shape)
        #print(y_z[0, -1, :])
        #print(y_z.size())
        
        # two clusters
        y_expanded = torch.cat([y_z, y_z], dim=1)
        #expand(y_z)
        #print(y_expanded.size())
        pdf_y = th.exp(N.log_prob(y_expanded))
        pdf_y = reshape(pdf_y)
        #print(y_expanded)
        #print(pdf_y)
        # sample z to build empirical sample mean over z for the likelihood
        # we are using only one sample at time from the mixture ----> likelihood
        # is simply the normal
        loglikelihood = 0
        # for every sample compute the weighted mixture
        for sample in range(self.n_samples_z):
            eps = V(th.randn(y_expanded.size()))
            # we use z_y as a selector/weight (z_i is a three dimensional Gaussian
            # in this way we can also measure uncertainly)
            z_y = eps * std_phi + mu_phi

            z_y = reshape(z_y)
            z_y = F.softmax(z_y, dim=2)
            # log of mixture weighted with z
            #print(z_y.shape)
            #print(pdf_y.shape)
            s = th.sum(pdf_y * z_y, dim=2)
            loglikelihood += th.log(th.sum(pdf_y * z_y, dim=2))

        loglikelihood /= self.n_samples_z
        loglikelihood = th.sum(loglikelihood, dim=1)
        loglikelihood = th.mean(loglikelihood) #/ y_z.size()[0]*y_z.size()[1]
        # reduce mean over the batch size reduce sum over the lidars
        if torch.cuda.is_available():
            loglikelihood = loglikelihood.cuda()
        
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
        return -elbo, kld, loglikelihood, pdf_y, z_y, s