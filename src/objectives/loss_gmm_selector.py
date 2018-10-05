import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

from utils.utils import move_to_cuda

class LossSelector(nn.Module):
    def __init__(self, batch_size=2, n_samples_y=10, input_size=9, n_clusters=2, lmbda=0.001, model_type="selector", is_entropy=False):
        super(LossSelector, self).__init__()
        '''
        add annealing alpha (1-alpha)
        convert in a class
        '''
        self.batch_size  = batch_size
        self.n_samples_y = n_samples_y
        
        self.input_size = input_size
        self.n_clusters = n_clusters

        self.lmbda = lmbda
        self.model_type  = model_type
        self.is_entropy = is_entropy

    def routine(self, x):
        x = x.reshape(-1, self.n_clusters, self.input_size).permute(0,2,1)
        return x

    def forward(self, y, mu_c, std_c, clusters):
        y = y.view(-1, self.n_samples_y, self.input_size)
        y = y[:,-1,:]
        
        if self.model_type == "gmm":
            y = th.cat([y, y], dim=1)
            y = self.routine(y)
        
        N = Normal(mu_c, std_c)
        pdf_y = N.log_prob(y)

        if self.model_type == "gmm":
            pdf_y  = th.sum(clusters*pdf_y,  dim=2)

        loglikelihood = th.sum(pdf_y, dim=1)
        entropy = th.sum(clusters * th.log(clusters), dim=2)
        entropy = th.sum(entropy, dim=1) 

        loss = - th.mean(loglikelihood)
        if self.is_entropy:
            loss = loss - self.lmbda * th.mean(entropy)
        return loss