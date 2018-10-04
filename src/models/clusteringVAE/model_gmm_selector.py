import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Normal
# we frame all the problem as an unsupervised learning problem.
# we want to use a Conditional Variational Auto-Encoder to tackle the problem.

# Y observable variables ----> 9 lidar measurements
# X conditioning variables ----> 7 joint angles + robot state
# Z hidden variables ----> distribution over possible classes self, agent, background
# Z is a multidimensional (9) mixture gaussians (3)

# at inference time, we have interest in q(z|y,x) 

# model assumption: 
# for every lidar (9) exists a (k) component clustering to cluster its measurements

class Encoder(nn.Module):
    def __init__(self, layer_sizes, latent_size, flag=True):

        super().__init__()
        self.flag = flag
        self.MLP = nn.Sequential()

        for i, (in_size, out_size) in enumerate( zip(layer_sizes[:-1], layer_sizes[1:]) ):
            self.MLP.add_module(name="L%i"%(i), module=nn.Linear(in_size, out_size))
            self.MLP.add_module(name="A%i"%(i), module=nn.Tanh())

        self.linear_means = nn.Linear(layer_sizes[-1], latent_size)
        if self.flag:
            self.linear_log_var = nn.Linear(layer_sizes[-1], latent_size)

    def forward(self, x):
        x = self.MLP(x)

        self.mu_phi = self.linear_means(x)

        if self.flag:
            self.log_var_phi = self.linear_log_var(x)
            return self.mu_phi, self.log_var_phi
        return self.mu_phi

class VAE(nn.Module):
    def __init__(self, 
                 encoder_layer_sizes,
                 latent_size,
                 n_clusters,
                 batch_size,
                 flag="selector"):

        super().__init__()
        assert type(encoder_layer_sizes) == list
        assert type(latent_size) == int
        # in our case the latent space is 9 (lidars) x 3 (hidden states)
        # for any lidar we assume a different mixture of gaussian
        
        # to sample z we need the means (9 x 3) and the log of the variance (9 x 3) of our multidimensional
        # mixture of gaussians
        self.latent_size = latent_size
        self.batch_size = batch_size
        self.n_clusters = n_clusters
        self.input_size = self.latent_size // self.n_clusters

        # the encoder builds deterministic moments for the approx posterior q_{phi}(z|y, x)
        # the encoder is typically a MLP
        self.encoder = Encoder(encoder_layer_sizes, latent_size, True)
        # the decoder compute the approx likelihood p_{theta}(y|z, x)
        self.cluster = Encoder(encoder_layer_sizes, latent_size, False)

        self.flag = flag
        
        self.init_parameters()

    def init_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def routine(self, x):
        x = x.reshape(-1, self.n_clusters, self.input_size).permute(0,2,1)
        return x

    def forward(self, x):
        mu_phi, log_var_phi = self.encoder(x)
        clusters = self.cluster(x)

        mu_phi  = self.routine(mu_phi)
        std_phi = th.exp(0.5 * log_var_phi)
        std_phi = self.routine(std_phi)

        clusters = self.routine(clusters)
        clusters = F.softmax(clusters, dim=2)

        if self.flag=="selector":
            mu_c  = th.sum(clusters * mu_phi,  dim=2)
            std_c = th.sum(clusters * std_phi, dim=2)
        elif self.flag=="gmm":
            mu_c  = mu_phi
            std_c = std_phi
        else:
            mu_c  = mu_phi
            std_c = std_phi
        return mu_c, std_c, clusters

    def inference(self, y, x, n=1):
        mu_phi, log_var_phi = self.encoder(x)
        std_phi = th.exp(0.5 * log_var_phi)

        N = Normal(mu_phi, std_phi)

        y_sample = N.sample()
        y_sample = y_sample.reshape(-1, self.n_clusters, y.size()[-1]).permute(0,2,1)
        y_sample = F.softmax(y_sample, dim=2)
        return y_sample

if __name__ == '__main__':
    vae = VAE(batch_size=24, encoder_layer_sizes=[7, 256, 256], latent_size=18, n_clusters=2)
    x = Variable(th.zeros(24, 7))
    x[0, 3] = 1
    y = Variable(th.zeros(24, 9))
    
    mu_phi, log_var_phi, clusters = vae(x)
    print(mu_phi.size())
    print(log_var_phi.size())

    print(mu_phi[0])

    print(mu_phi[0].reshape(-1, 2, 9).permute(0,2,1))

    print(vae.inference(y, x).shape)