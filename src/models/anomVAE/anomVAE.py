from __future__ import print_function
import argparse
import torch as th
from torch import nn, optim
from torch.nn import functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from src.loaders.load_panda import PandaDataSet
import os.path as osp
from src.utils.utils import path_exists
from utils import configs as cfg

############################################################
### INITIALIZATION
############################################################

parser = argparse.ArgumentParser(description='VAE Anomaly Example')
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--code_size', type=int, default=20, metavar='N',
                    help='code size (default: 20)')
parser.add_argument('--lr', type=float, default=1e-3, metavar='N',
                    help='learning rate (default: 1e-3)')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
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

        self.fc1 = nn.Linear(768, 400)
        self.fc21 = nn.Linear(400, args.code_size)
        self.fc22 = nn.Linear(400, args.code_size)
        self.fc3 = nn.Linear(args.code_size, 400)
        self.fc4 = nn.Linear(400, 768)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = th.exp(0.5*logvar)
            eps = th.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return F.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 768))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr)

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 768), size_average=False)

    # see Appendix B from Generative_Models paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * th.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))
    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


# def test(epoch):
#     model.eval()
#     test_loss = 0
#     with th.no_grad():
#         for i, (data, _) in enumerate(test_loader):
#             data = data.to(device)
#             recon_batch, mu, logvar = model(data)
#             test_loss += loss_function(recon_batch, data, mu, logvar).item()
#             if i == 0:
#                 n = min(data.size(0), 8)
#                 comparison = th.cat([data[:n],
#                                       recon_batch.view(-1, 1, 32, 24)[:n]])
#                 save_image(comparison.cpu(), os.path.join(path_results, 'reconstruction_' + str(epoch) + '.png'), nrow=n)
# 
#     test_loss /= len(test_loader.dataset)
#     print('====> Test set loss: {:.4f}'.format(test_loss))


for epoch in range(1, args.epochs + 1):
    train(epoch)
    