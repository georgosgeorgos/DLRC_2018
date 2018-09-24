import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils import *
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
    def __init__(self, layer_sizes, latent_size, conditional, num_labels):

        super().__init__()

        self.conditional = conditional
        if self.conditional:
            layer_sizes[0] += num_labels

        self.MLP = nn.Sequential()

        for i, (in_size, out_size) in enumerate( zip(layer_sizes[:-1], layer_sizes[1:]) ):
            self.MLP.add_module(name="L%i"%(i), module=nn.Linear(in_size, out_size))
            self.MLP.add_module(name="A%i"%(i), module=nn.Tanh())

        self.linear_means   = nn.Linear(layer_sizes[-1], latent_size)
        self.linear_log_var = nn.Linear(layer_sizes[-1], latent_size)

    def forward(self, y, x=None):
        if self.conditional:
            y = th.cat((y, x), dim=-1)

        y = self.MLP(y)

        self.mu_phi      = self.linear_means(y)
        self.log_var_phi = self.linear_log_var(y)
        return self.mu_phi, self.log_var_phi


class Decoder(nn.Module):
    def __init__(self, latent_size, layer_sizes, batch_size, conditional, num_labels):

        super().__init__()

        self.MLP = nn.Sequential()

        self.conditional = conditional
        if self.conditional:
            input_size = latent_size + num_labels
        else:
            input_size = latent_size

        for i, (in_size, out_size) in enumerate( zip(layer_sizes[:-1], layer_sizes[1:]) ):
            self.MLP.add_module(name="L%i"%(i), module=nn.Linear(in_size, out_size))
            self.MLP.add_module(name="A%i"%(i), module=nn.Tanh())

        self.linear_means   = nn.Linear(layer_sizes[-1], latent_size)
        self.linear_log_var = nn.Linear(layer_sizes[-1], latent_size)

        self.mu_theta      = V(th.zeros((batch_size, latent_size)))
        self.log_var_theta = V(th.zeros((batch_size, latent_size)))

    def forward(self, z=None, x=None):
        if self.conditional:
            x = self.MLP(x)
            self.mu_theta      = self.linear_means(y)
            self.log_var_theta = self.linear_log_var(y)

        return self.mu_theta, self.log_var_theta


class VAE(nn.Module):
    def __init__(self, 
                 encoder_layer_sizes,
                 decoder_layer_sizes,
                 latent_size,
                 batch_size,  
                 conditional=False, 
                 num_labels=0):

        super().__init__()

        if conditional:
            assert num_labels > 0

        assert type(encoder_layer_sizes) == list
        assert type(latent_size) == int
        # in our case the latent space is 9 (lidars) x 3 (hidden states)
        # for any lidar we assume a different mixture of gaussian
        
        # to sample z we need the means (9 x 3) and the log of the variance (9 x 3) of our multidimensional
        # mixture of gaussians
        self.latent_size = latent_size

        # the encoder builds deterministic moments for the approx posterior q_{phi}(z|y, x)
        # the encoder is typically a MLP
        self.encoder = Encoder(encoder_layer_sizes, latent_size, conditional, num_labels)
        # the decoder compute the approx likelihood p_{theta}(y|z, x)
        self.decoder = Decoder(latent_size, decoder_layer_sizes, batch_size, conditional, num_labels)

        self.init_parameters()

    def init_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, y, x=None):
        mu_phi, log_var_phi     = self.encoder(y, x)
        mu_theta, log_var_theta = self.decoder(x)

        return mu_phi, log_var_phi, mu_theta, log_var_theta

    def inference(self, y, n=1, x=None):
        '''
        at inference time, given the lidar measure, we want a distribution
        over classes z|y,x

        '''
        batch_size = n
        # KL divergence matches the z distribution with N(0, 1)
        #z = v(th.randn([batch_size, self.latent_size]))
        mu_phi, log_var_phi = self.encoder(y, x)
        std_phi = std(log_var_phi)

        ## assuming a normal model for z|y,x
        eps = V(th.randn([batch_size, self.latent_size]))
        ## z is 18 dimensional (9 lidars x 2 states)
        z_y = eps * std_phi + mu_phi
        #print(z_y)

        z_y = reshape(z_y)
        #print(z_y)
        z_y = F.softmax(z_y, dim=2)

        return z_y

if __name__ == '__main__':
    vae = VAE(batch_size=24, conditional=False, num_labels=7)
    y = Variable(th.zeros(24, 9))
    x = Variable(th.zeros(24, 7))

    mu_phi, log_var_phi, mu_theta, log_var_theta = vae(y)
    print(mu_phi.size())
    print(log_var_phi.size())
    print(mu_theta.size())
    print(log_var_theta.size())
    mu_phi, log_var_phi, mu_theta, log_var_theta = vae(y, x)
    print(mu_phi.size())
    print(log_var_phi.size())
    print(mu_theta.size())
    print(log_var_theta.size())

    z_y = vae.inference(y)
    print(z_y)
    print(z_y.size())
    print(z_y.sum(2))

    def loss_fn(y_z, mu_phi, log_var_phi, mu_theta, log_var_theta, batch_size):
        from torch.distributions.normal import Normal

        loglikelihood = 0
        
        std_theta = std(log_var_theta)
        std_phi   = std(log_var_phi)
        
        N = Normal(mu_theta, std_theta)
        #y_z = th.unsqueeze(y, 2)
        #print(y_z.size())

        y_expanded = th.cat([y_z, y_z, y_z], dim=1)
        #print(y_expanded.size())
        pdf_y = th.exp(N.log_prob(y_expanded))
        pdf_y = reshape(pdf_y)
        #print(pdf_y.size())
        # sample z
        for sample in range(10):
            eps = V(th.randn(y_expanded.size()))
            z_y = eps * std_phi + mu_phi

            z_y = reshape(z_y)
            z_y = F.softmax(z_y, dim=2)
            loglikelihood += th.log(th.sum(pdf_y * z_y, dim=2))

        #print(loglikelihood.size())
        loglikelihood /= 10
        loglikelihood = th.sum(loglikelihood) / y_z.size()[0]*y_z.size()[1]
        # reduce mean over the batch size reduce sum over the lidars
        
        # reduce over KLD
        kld = 1/2 * torch.sum(log_var_phi.exp() + mu_phi.pow(2) - 1 - log_var_phi)

        print(kld)
        print(loglikelihood)
        
        elbo = loglikelihood - kld 

        return - elbo


    l = loss_fn(y, mu_phi, log_var_phi, mu_theta, log_var_theta)
    print(l)
