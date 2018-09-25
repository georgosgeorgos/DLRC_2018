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

############################################################
### INITIALIZATION
############################################################

lr = 1e-5
every_nth = 100
trbs = 128
epochs = 10
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

        self.fc1 = nn.Linear(7, 128)  # num_joints x num_hidden
        self.fc2 = nn.Linear(128, 128)

        self.mu_layer = nn.Linear(128, 9)  # num_hidden x num_channels
        self.logvar_layer = nn.Linear(128, 9)

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
    for batch_idx, (x,y) in enumerate(train_loader):
        n, m = x.size()
        x.resize_(min(trbs, n), m)
        x = x.to(device).float()
        optimizer.zero_grad()
        mu, logvar = model(x)
        loss = loss_fn(mu, logvar, y)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % every_nth == 0:
            print("train epoch: {}/{} [{}/{} ({:.0f}%)]".format(
                epoch, epochs, (batch_idx + 1) * trbs - (trbs - x.size()[0]),
                len(train_loader.dataset), 100. * (batch_idx + 1) / len(train_loader)))

    print('====> epoch: {} Average loss: {:.4f}\n'.format(
          epoch, train_loss / len(train_loader.dataset)))

############################################################
### EXECUTE MODEL
############################################################


if __name__ == '__main__':
    for epoch in range(1, epochs + 1):
        train(epoch)