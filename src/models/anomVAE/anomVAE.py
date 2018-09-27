from __future__ import print_function
import argparse
import torch as th
from torch import nn, optim
from torch.nn import functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from src.loaders.load_panda import PandaDataSet
import os.path as osp
from src.utils.utils import path_exists, plot_eval
from utils import configs as cfg
import numpy as np

############################################################
### INITIALIZATION
############################################################

parser = argparse.ArgumentParser(description='VAE Anomaly Example')
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--code_size', type=int, default=10, metavar='N',
                    help='code size (default: 2)')
parser.add_argument('--lr', type=float, default=1e-4, metavar='N',
                    help='learning rate (default: 1e-3)')
parser.add_argument('--epochs', type=int, default=300, metavar='N',
                    help='number of epochs to train (default: 150)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and th.cuda.is_available()

device = th.device("cuda" if args.cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

test_every_nth = 1
n_joints = 7
n_lidars = 9
n_hidden = 128
verbose = False
trbs = 512
tebs = 256
rootdir = '../../../'
path_results = osp.join(rootdir, 'experiments', 'anomVAE')
path_exists(path_results)

train_set = PandaDataSet(root_dir=osp.join(rootdir, 'data/data_toy'), train=True,
                             transform=transforms.Compose([
            transforms.Lambda(lambda n: th.Tensor(n)),
            transforms.Lambda(lambda n: th.Tensor.clamp(n, cfg.LIDAR_MIN_RANGE, cfg.LIDAR_MAX_RANGE)),
            transforms.Lambda(lambda n: n / 1000)
        ])
                             )
train_loader = DataLoader(train_set, batch_size=trbs, shuffle=True, **kwargs)

test_set = PandaDataSet(root_dir=osp.join(rootdir, 'data/data_toy'), train=False,
                         transform=transforms.Compose([
                             transforms.Lambda(lambda n: th.Tensor(n)),
                             transforms.Lambda(
                                 lambda n: th.Tensor.clamp(n, cfg.LIDAR_MIN_RANGE, cfg.LIDAR_MAX_RANGE)),
                             transforms.Lambda(lambda n: n / 1000)
                         ])
                         )
test_loader = DataLoader(test_set, batch_size=tebs, shuffle=True, **kwargs)


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(n_lidars+2*n_joints, n_hidden)
        self.fc21 = nn.Linear(n_hidden, args.code_size)
        self.fc22 = nn.Linear(n_hidden, args.code_size)
        self.fc3 = nn.Linear(args.code_size, n_hidden)
        self.fc4 = nn.Linear(n_hidden, n_lidars+2*n_joints)

    def encode(self, *input):
        h1 = th.tanh(self.fc1(input[0]))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = th.exp(0.5*logvar)
            eps = th.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        h3 = th.tanh(self.fc3(z))
        return th.tanh(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, n_lidars+2*n_joints))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.mse_loss(recon_x, x.view(-1, n_lidars+2*n_joints), reduction='sum')

    # see Appendix B from Generative_Models paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * th.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


def train(epoch):
    model.train()
    train_loss = 0.
    for batch_idx, (x, y, z) in enumerate(train_loader):
        x = x.to(device).float()
        y = y.to(device).float()
        z = z.to(device).float()

        input_concat = th.cat((x, y, z), dim=1)

        optimizer.zero_grad()
        recon_batch, mu, logvar = model(input_concat)
        loss = loss_function(recon_batch, input_concat, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if verbose:
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(input_concat), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss.item() / len(input_concat)))
    epoch_loss = train_loss / len(train_loader.dataset)
    print('train epoch: {} avg. loss: {:.4f}'.format(epoch, epoch_loss))

    return epoch_loss


def test(epoch):

    model.eval()
    test_loss = 0.
    MSE = 0.

    with th.no_grad():
        for i, (x, y, z) in enumerate(test_loader):
            bs, _, = y.size()

            x = x.to(device).float()
            y = y.to(device).float()
            z = z.to(device).float()

            input_concat = th.cat((x, y, z), dim=1)

            input_recon, mu, logvar = model(input_concat)
            test_loss += loss_function(input_recon, input_concat, mu, logvar).item()

            MSE += F.mse_loss(input_concat, input_recon, reduction='elementwise_mean')

    epoch_loss = test_loss / len(test_loader.dataset)
    epoch_RMSE = th.sqrt(MSE / len(test_loader.dataset))
    print(' test epoch: {} avg. loss: {:.4f}\tRMSE: {:.4f}\n'.format(epoch, epoch_loss, epoch_RMSE))

    return epoch_loss, epoch_RMSE


if __name__ == '__main__':

    train_loss_history = []

    test_loss_history = []
    test_RMSE_history = []

    for epoch in range(1, args.epochs + 1):
        train_loss_history.append(train(epoch))

        if epoch % test_every_nth == 0:
            epoch_loss, epoch_RMSE = test(epoch)
            test_loss_history.append(epoch_loss)
            test_RMSE_history.append(epoch_RMSE)

    plot_eval(np.arange(len(train_loss_history)), np.array(train_loss_history), xlabel='epochs', ylabel='loss',
              title='train loss', save_to=osp.join(path_results, 'train_loss.png'))
    plot_eval(np.arange(len(test_loss_history)), np.array(test_loss_history), xlabel='epochs', ylabel='loss',
              title='test loss', save_to=osp.join(path_results, 'test_loss.png'))
    plot_eval(np.arange(len(test_RMSE_history)), np.array(test_RMSE_history), xlabel='epochs', ylabel='RMSE',
              title='test RMSE', save_to=osp.join(path_results, 'test_RMSE.png'))