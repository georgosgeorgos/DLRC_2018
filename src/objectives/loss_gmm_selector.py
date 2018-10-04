import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

from utils.utils import move_to_cuda

class LossSelector(nn.Module):
    def __init__(self, batch_size=2, n_samples_y=10, input_size=9, n_clusters=2, flag="selector", is_entropy=False):
        super(LossSelector, self).__init__()
        '''
        add annealing alpha (1-alpha)
        convert in a class
        '''
        self.batch_size  = batch_size
        self.n_samples_y = n_samples_y
        
        self.input_size = input_size
        self.n_clusters = n_clusters

        self.flag = flag
        self.is_entropy = is_entropy

    def forward(self, y, mu_c, std_c, clusters):
        y = y.view(-1, self.n_samples_y, self.input_size)
        y = y[:,-1,:]

        if self.flag == "gmm":
            y = th.cat([y, y], dim=1)
        
        N = Normal(mu_c, std_c)
        pdf_y = N.log_prob(y)

        if self.flag == "gmm":
            pdf_y  = th.sum(pdf_y,  dim=2)

        loglikelihood = th.sum(pdf_y, dim=1)
        entropy = th.sum(clusters * th.log(clusters), dim=2)
        entropy = th.sum(entropy, dim=1) 

        loss = - th.mean(loglikelihood)
        if is_entropy:
            loss = loss - th.mean(entropy)
        return loss