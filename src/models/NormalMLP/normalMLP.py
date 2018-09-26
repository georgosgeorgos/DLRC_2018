import torch as th
from torch.nn import functional as F
from torch import nn, optim
from torchvision import transforms
from src.objectives.llnormal import LLNormal
from src.loaders.load_panda import PandaDataSet
from torch.utils.data import DataLoader
from torch.autograd import Variable
import utils.configs as cfg
from utils.utils import plot_eval, path_exists, plot_hist
import numpy as np
import os.path as osp

############################################################
### INITIALIZATION
############################################################

n_joints = 7
n_lidars = 9
n_hidden = 128
verbose = False
lr = 1e-3
every_nth = 100
trbs = 512
tebs = 256
epochs = 200
test_every_nth = 1
th.manual_seed(42)
cuda = th.cuda.is_available()
device = th.device("cuda" if cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

rootdir = '../../../'
train_set = PandaDataSet(root_dir=osp.join(rootdir, 'data/data_toy'), train=True,
                             transform=transforms.Compose([
            transforms.Lambda(lambda n: th.Tensor(n)),
            transforms.Lambda(lambda n: th.Tensor.clamp(n, cfg.LIDAR_MIN_RANGE, cfg.LIDAR_MAX_RANGE)),
            transforms.Lambda(lambda n: n / 1000)
        ])
                             )
train_loader = DataLoader(train_set, batch_size=trbs, shuffle=True, **kwargs)

test_set = PandaDataSet(root_dir='../../../data/data_toy', train=False,
                         transform=transforms.Compose([
                             transforms.Lambda(lambda n: th.Tensor(n)),
                             transforms.Lambda(
                                 lambda n: th.Tensor.clamp(n, cfg.LIDAR_MIN_RANGE, cfg.LIDAR_MAX_RANGE)),
                             transforms.Lambda(lambda n: n / 1000)
                         ])
                         )
test_loader = DataLoader(test_set, batch_size=tebs, shuffle=True, **kwargs)

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

        h1 = th.tanh(self.fc1(input[0]))
        h2 = th.tanh(self.fc2(h1))

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
        x = Variable(x)
        x = x.to(device).float()
        y = Variable(y)
        y = y.to(device).float()

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

    epoch_loss = train_loss / len(train_loader.dataset)
    print('====> train epoch: {} avg. loss: {:.4f}'.format(epoch, epoch_loss))
    return epoch_loss


def test(epoch):

    model.eval()
    test_loss = 0.
    y_cen = []

    with th.no_grad():
        for i, (x, y) in enumerate(test_loader):
            n, m = x.size()
            x.resize_(min(tebs, n), m)
            x = Variable(x)
            x = x.to(device).float()
            y = Variable(y)
            y = y.to(device).float()

            mu, logvar = model(x)
            std = th.exp(0.5 * logvar)

            loss = loss_fn(mu, logvar, y)
            test_loss += loss.item()

            y_cen_batch = (y.cpu().data.numpy() - mu.cpu().data.numpy()) / std.cpu().data.numpy()
            y_cen_batch = y_cen_batch.tolist()
            y_cen.append(y_cen_batch)

    y_cen_array = np.array(y_cen[:len(y_cen)-1])  # exclude last batch which might have different size than `tebs`
    y_cen_array = np.reshape(y_cen_array, (y_cen_array.shape[0]*y_cen_array.shape[1], n_lidars))  # reshape to [num_samples x num_channels]
    y_cen_array = np.vstack((y_cen_array, np.array(y_cen[-1])))  # add last batch

    epoch_loss = test_loss / len(test_loader.dataset)
    print('### TEST: epoch: {} avg. loss: {:.4f}\n'.format(epoch, epoch_loss))
    return epoch_loss, y_cen_array

############################################################
### EXECUTE MODEL
############################################################


if __name__ == '__main__':

    path_results = osp.join(rootdir, 'experiments', 'normalMLP')
    path_exists(path_results)
    train_loss_history = []
    test_loss_history = []

    for epoch in range(1, epochs + 1):
        train_loss_history.append(train(epoch))

        if epoch % test_every_nth == 0:
            epoch_loss, hist_values = test(epoch)
            test_loss_history.append(epoch_loss)

    plot_eval(np.arange(epochs), np.array(train_loss_history),
              save_to=osp.join(path_results, 'train_loss.png'), title='train loss')
    plot_eval(np.arange(epochs), np.array(test_loss_history),
              save_to=osp.join(path_results, 'test_loss.png'), title='test loss')
    plot_hist(hist_values,
              save_to=osp.join(path_results, 'test_histogram.png'), title='Histograms of target measurements (on test set) for each channel')