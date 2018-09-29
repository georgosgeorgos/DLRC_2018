import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

from utils.utils import move_to_cuda

class LossSelector(nn.Module):
    def __init__(self, batch_size=2, n_samples_y=10, input_size=9, n_clusters=2):
        super(LossSelector, self).__init__()
        '''
        add annealing alpha (1-alpha)
        convert in a class
        '''
        self.batch_size  = batch_size
        self.n_samples_y = n_samples_y
        
        self.input_size = input_size
        self.n_clusters  = n_clusters

    def forward(self, y, mu_c, std_c, clusters):
        y = y.view(-1, self.n_samples_y, self.input_size)
        y = y[:,-1,:]
        
        N = Normal(mu_c, std_c)
        pdf_y = N.log_prob(y)

        loglikelihood = th.sum(pdf_y, dim=1)

        entropy = th.sum(clusters * th.log(clusters), dim=2)
        entropy = th.sum(entropy, dim=1) 

        loss = - th.mean(loglikelihood + entropy)
        return loss