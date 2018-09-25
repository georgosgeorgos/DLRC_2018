import torch.nn as nn
import torch as th
import torch.nn.functional as F
from torch.distributions.normal import Normal


class LLNormal(nn.Module):
    """
    L(Y, X) &= \prod_i p(y_i \vert x_i) \\
            & = \prod_i\prod_jN(
            y_{ij}^{lidar} \vert \mu_{j}^{lidar}(x_i), \sigma_j^{lidar}(x_i)^2)
    """
    def __init__(self):
        super(LLNormal, self).__init__()

    def forward(self, *input):

        mu_phi = input[0]
        logvar_phi = input[1]
        y = input[2]  # lidar measurements
        std_phi = th.exp(0.5 * logvar_phi)

        N = Normal(mu_phi, std_phi)
        loglikelihood = th.sum(th.exp(N.log_prob(y)))

        return loglikelihood


# if __name__ == '__main__':
#     loss = LLNormal()
#     mu = th.randn(1, 9, requires_grad=True)
#     var = th.randn(1, 9, requires_grad=True)
#     Y = th.randn(1, 9)
#     print(mu, var, Y)
#     output = loss(mu, var, Y)
#     print(output)
