from __future__ import print_function

import argparse
import os.path as osp

import numpy as np
import torch as th
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from src.loaders.load_panda import PandaDataSet
from src.utils.utils import path_exists, plot_eval, cumulative_moving_average, plot_hist, plot_correlation_matrix, \
    ckpt_utc, plot_scatter
from src.utils import configs as cfg

############################################################
### INITIALIZATION
############################################################

parser = argparse.ArgumentParser(description='VAE for Anomaly Detection')
parser.add_argument('--trbs', type=int, default=256, metavar='N',
                    help='input batch size for training (default: 256)')
parser.add_argument('--tebs', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--n_lidars', type=int, default=9, metavar='N',
                    help='lidar size (default: 9)')
parser.add_argument('--n_joints', type=int, default=7, metavar='N',
                    help='joint size (default: 7)')
parser.add_argument('--n_hidden', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--code_size', type=int, default=2, metavar='N',
                    help='code size (default: 2)')
parser.add_argument('--lr', type=float, default=1e-3, metavar='N',
                    help='learning rate (default: 1e-3)')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 150)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--log-interval', type=int, default=5, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--verbose', type=int, default=0,
                    help='log in verbose mode?')
parser.add_argument('--plot_every_nth', type=int, default=10, metavar='N',
                    help='every how many epochs to plot during training')
parser.add_argument('--test_every_nth', type=int, default=1, metavar='N',
                    help='every how many epochs to test')
parser.add_argument('--rootdir', type=str, default='../../../',
                    help='location of project root')

args = parser.parse_args()
args.cuda = not args.no_cuda and th.cuda.is_available()

device = th.device("cuda" if args.cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

train_set = PandaDataSet(root_dir=osp.join(args.rootdir, 'data'), filename='train_data_correct.pkl', train=True,
                         transform=transforms.Compose([
                             transforms.Lambda(lambda n: th.Tensor(n)),
                             transforms.Lambda(lambda n: th.Tensor.clamp(n, cfg.LIDAR_MIN_RANGE, cfg.LIDAR_MAX_RANGE)),
                             transforms.Lambda(lambda n: n / 1000)
                         ]))
train_loader = DataLoader(train_set, batch_size=args.trbs, shuffle=True, **kwargs)

test_set = PandaDataSet(root_dir=osp.join(args.rootdir, 'data'), filename='train_data_correct.pkl', train=False,
                        transform=transforms.Compose([
                            transforms.Lambda(lambda n: th.Tensor(n)),
                            transforms.Lambda(
                                lambda n: th.Tensor.clamp(n, cfg.LIDAR_MIN_RANGE, cfg.LIDAR_MAX_RANGE)),
                            transforms.Lambda(lambda n: n / 1000)
                        ]))
test_loader = DataLoader(test_set, batch_size=args.tebs, shuffle=True, **kwargs)


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(args.n_lidars + 2 * args.n_joints, args.n_hidden)
        self.fc21 = nn.Linear(args.n_hidden, args.code_size)
        self.fc22 = nn.Linear(args.n_hidden, args.code_size)
        self.fc3 = nn.Linear(args.code_size, args.n_hidden)
        self.fc4 = nn.Linear(args.n_hidden, args.n_lidars + 2 * args.n_joints)

    def encode(self, *input):
        h1 = th.tanh(self.fc1(input[0]))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = th.exp(0.5 * logvar)
            eps = th.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        h3 = th.tanh(self.fc3(z))
        return th.tanh(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, args.n_lidars + 2 * args.n_joints))
        z = self.reparameterize(mu, logvar)
        return z, self.decode(z), mu, logvar


model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.mse_loss(recon_x, x.view(-1, args.n_lidars + 2 * args.n_joints), reduction='sum')

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
        latent_batch, recon_batch, mu, logvar = model(input_concat)
        loss = loss_function(recon_batch, input_concat, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if args.verbose:
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
    moving_avg_mean_latent = th.zeros(args.code_size)
    moving_avg_mean_latent = moving_avg_mean_latent.to(device)
    test_latent = []
    test_norm_target = []
    test_rel_err = []

    with th.no_grad():
        for i, (x, y, z) in enumerate(test_loader):
            bs, _, = y.size()

            x = x.to(device).float()
            y = y.to(device).float()
            z = z.to(device).float()

            input_concat = th.cat((x, y, z), dim=1)

            latent, input_recon, mu, logvar = model(input_concat)
            test_loss += loss_function(input_recon, input_concat, mu, logvar).item()

            MSE += F.mse_loss(input_concat, input_recon, reduction='elementwise_mean')

            # compute running means over latents
            new_latent_mean = th.mean(latent, dim=0)
            moving_avg_mean_latent = cumulative_moving_average(moving_avg_mean_latent, new_latent_mean, i)

            # compute norms
            norm_target = th.norm(input_concat, p=2, dim=1)
            norm_rel_err = th.norm((input_concat - input_recon), p=2, dim=1) / norm_target

            # accumulate latents
            test_latent.append(latent)

            # accumulate norm reconstructions and relative errors
            test_norm_target.append(norm_target)
            test_rel_err.append(norm_rel_err)

    epoch_loss = test_loss / len(test_loader.dataset)
    epoch_RMSE = th.sqrt(MSE / len(test_loader.dataset))
    print(' test epoch: {} avg. loss: {:.4f}\tRMSE: {:.4f}\n'.format(epoch, epoch_loss, epoch_RMSE))

    return epoch_loss, epoch_RMSE, moving_avg_mean_latent, latent, test_norm_target, test_rel_err


if __name__ == '__main__':

    path_results = osp.join(args.rootdir, 'experiments', 'anomVAE')
    path_exists(path_results)
    ckpt = ckpt_utc()
    ckpt_dir = osp.join(path_results, 'ckpt/')
    path_exists(ckpt_dir)

    train_loss_history = []

    test_loss_history = []
    test_RMSE_history = []

    for epoch in range(1, args.epochs + 1):
        train_loss_history.append(train(epoch))

        if epoch % args.test_every_nth == 0:
            epoch_loss, epoch_RMSE, moving_avg_mean_latent, latent, norm_target, rel_err = test(epoch)
            test_loss_history.append(epoch_loss)
            test_RMSE_history.append(epoch_RMSE)

        if epoch % args.plot_every_nth == 0:
            plot_eval(np.arange(len(train_loss_history)), np.array(train_loss_history), xlabel='epochs', ylabel='loss',
                      title='train loss', save_to=osp.join(path_results, 'train_loss.png'))
            plot_eval(np.arange(len(test_loss_history)), np.array(test_loss_history), xlabel='epochs', ylabel='loss',
                      title='test loss', save_to=osp.join(path_results, 'test_loss.png'))
            plot_eval(np.arange(len(test_RMSE_history)), np.array(test_RMSE_history), xlabel='epochs', ylabel='RMSE',
                      title='test RMSE', save_to=osp.join(path_results, 'test_RMSE.png'))
            plot_hist(moving_avg_mean_latent.cpu().data.numpy(), xlabel='value', ylabel='frequency',
                      title='test_hist_means_latent', save_to=osp.join(path_results, 'test_hist_means_latent.png'))

            # plot correlation matrix latents
            latent_concat = th.stack(tuple(latent), dim=0)
            plot_correlation_matrix(latent.cpu().data.numpy(), title='test_corr_matrix_latent',
                                    save_to=osp.join(path_results, 'test_corr_matrix_latent.png'))

            # plot norm target vs relative error
            norm_target_concat = th.cat(tuple(norm_target), dim=0)
            norm_rel_err_concat = th.cat(tuple(rel_err), dim=0)
            plot_scatter(y=norm_rel_err_concat.cpu().data.numpy(), x=norm_target_concat.cpu().data.numpy(),
                         ylabel="relative error (norm(target - recon) / norm(target))", xlabel="norm(target)",
                         title="norm vs rel err",
                         save_to=osp.join(path_results, 'test_norm_rel_err.png'))

    th.save(model.state_dict(), osp.join(ckpt_dir, ckpt))
