import torch as th
import torch.nn as nn
from torch.nn import functional as F
from torch import nn, optim
from torchvision import transforms
from objectives.llnormal import LLNormal
from data_loaders.load_panda import PandaDataSet
from torch.utils.data import DataLoader
import configs as cfg
import sys
from torch.distributions.normal import Normal

############################################################
### INITIALIZATION
############################################################

n_joints = 7
n_lidars = 9
n_hidden = 128
verbose = False
lr = 1e-4
every_nth = 100
trbs = 512
epochs = 1000
th.manual_seed(42)
cuda = th.cuda.is_available()
device = th.device("cuda" if cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

ds = PandaDataSet(
        transform=transforms.Compose([
            transforms.Lambda(lambda n: th.Tensor(n)),
            transforms.Lambda(lambda n: th.Tensor.clamp(n, cfg.LIDAR_MIN_RANGE, cfg.LIDAR_MAX_RANGE)),
            transforms.Lambda(lambda n: n / 1000)
        ])
    )
train_loader = DataLoader(ds, batch_size=trbs, shuffle=True, **kwargs)

############################################################
### MODEL
############################################################


class NormalMLP(nn.Module):

    def __init__(self):
        super(NormalMLP, self).__init__()

        self.fc1 = nn.Linear(n_joints, n_hidden)  # num_joints x num_hidden
        self.fc2 = nn.Linear(n_hidden, n_hidden)

        self.mu_layer = nn.Linear(n_hidden, n_lidars)  # num_hidden x num_channels
        self.logvar_layer = nn.Linear(n_hidden, n_lidars)

    def forward(self, *input):

        h1 = F.tanh(self.fc1(input[0]))
        h2 = F.tanh(self.fc2(h1))

        return self.mu_layer(h2), self.logvar_layer(h2)


model = NormalMLP().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
loss_fn = LLNormal()


############################################################
### HELPER FUNCTIONS
############################################################

def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (x, y) in enumerate(train_loader):

        n, m = x.size()
        x.resize_(min(trbs, n), m)
        x = x.to(device).float()

        optimizer.zero_grad()
        mu, logvar = model(x)

        loss = loss_fn(mu, logvar, y)
        loss.backward()

        train_loss += loss.item()

        optimizer.step()

        if verbose:
            if batch_idx % every_nth == 0:
                print("train epoch: {}/{} [{}/{} ({:.0f}%)]".format(
                    epoch, epochs, (batch_idx + 1) * trbs - (trbs - x.size()[0]),
                    len(train_loader.dataset), 100. * (batch_idx + 1) / len(train_loader)))

    print(' ---> epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))


def test(epoch):

    model.eval()
    test_loss = 0.

    with th.no_grad():
        for i, (x, y) in enumerate(train_loader):

            n, m = x.size()
            x.resize_(min(trbs, n), m)
            x = x.to(device).float()

            mu, logvar = model(x)
            N = Normal(mu, th.exp(0.5 * logvar))

            # # average prediction over 100 samples
            # y_pred = N.sample(sample_shape=(100,))
            # y_pred = th.mean(y_pred, dim=0)

            # test_loss += F.mse_loss(y_pred, y)

    print('====> test epoch: {} Average loss: {:.4f}\n'.format(epoch, test_loss / len(train_loader.dataset)))

############################################################
### EXECUTE MODEL
############################################################


if __name__ == '__main__':
    for epoch in range(1, epochs + 1):
        train(epoch)
        test(epoch)