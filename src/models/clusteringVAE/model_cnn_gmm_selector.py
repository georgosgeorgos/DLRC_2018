import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Normal
import sys
sys.path.append('/home/georgos/DLRC_2018/')
from src.models.autoencoder.autoencoder import Autoencoder

class EncoderCNN(nn.Module):
    def __init__(self,  
                 latent_size=18, 
                 is_uq=True):

        super().__init__()
        self.latent_size = latent_size
        self.is_uq = is_uq
        self.CNN   = Autoencoder()
        self.pool  = nn.MaxPool2d(4, stride=2)
        self.encoder_layer = nn.Conv2d(256, 256, 5, stride=2)

    def forward(self, x):
        _, x_compress = self.CNN(x)
        x_compress = self.encoder_layer(self.pool(x_compress))
        x_compress = x_compress.flatten()

        self.mu_phi = nn.Linear(x_compress.size()[0], self.latent_size)(x_compress)
        
        if self.is_uq:
            self.log_var_phi = nn.Linear(x_compress.size()[0], self.latent_size)(x_compress)
            return self.mu_phi, self.log_var_phi

        return self.mu_phi