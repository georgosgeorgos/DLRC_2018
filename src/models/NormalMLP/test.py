import numpy as np
import torch as th
import glob
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data import DataLoader
import torch as th
from loaders.load_panda import PandaDataSet
from src.models.NormalMLP.normalMLP import NormalMLP
import os.path as osp
from src.utils.utils import path_exists, ckpt_utc, plot_timeseries
from src.utils import configs as cfg
from torch.distributions.normal import Normal

tebs = 1
th.manual_seed(42)
cuda = th.cuda.is_available()
device = th.device("cuda" if cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
rootdir = '../../../'
path_results = osp.join(rootdir, 'experiments', 'normalMLP')
path_exists(path_results)
ckpt = ckpt_utc()
ckpt_dir = osp.join(path_results, 'ckpt/')
path_exists(ckpt_dir)

test_set = PandaDataSet(root_dir=osp.join(rootdir, 'data/data_toy'), train=False,
                         transform=transforms.Compose([
                             transforms.Lambda(lambda n: th.Tensor(n)),
                             transforms.Lambda(
                                 lambda n: th.Tensor.clamp(n, cfg.LIDAR_MIN_RANGE, cfg.LIDAR_MAX_RANGE)),
                             transforms.Lambda(lambda n: n / 1000)
                         ])
                         )
test_loader = DataLoader(test_set, batch_size=tebs, shuffle=False, **kwargs)

model = NormalMLP().to(device)

saved_state_dict = th.load(glob.glob(osp.join(ckpt_dir, "*.pth"))[0])
model.load_state_dict(saved_state_dict)

model.eval()

data = {}

for i, (x, y, _) in enumerate(test_loader):

    x = x.to(device).float()
    y = y.to(device).float()

    mu, logvar = model(x)
    std = th.exp(0.5 * logvar)
    N = Normal(mu, std)
    y_sample = N.sample((tebs,))

    y = y.cpu().data.numpy().tolist()
    mu = mu.cpu().data.numpy().tolist()
    std = std.cpu().data.numpy().tolist()
    pred = y_sample.cpu().data.numpy().tolist()

    data[i] = {"input": y, "pred": pred, "mu": mu, "std": std}

# Preprocess
input_list = []
pred_list  = []
mu_list = []
std_list = []

for k in range(len(data)):
    input_list.append(data[k]["input"])
    pred_list.append(data[k]["pred"])
    mu_list.append(data[k]["mu"])
    std_list.append(data[k]["std"])

input_array = np.array(input_list).squeeze()
pred_array = np.array(pred_list).squeeze()
mu_array = np.array(mu_list).squeeze()
std_array = np.array(std_list).squeeze()

plot_timeseries(input_array, pred_array, mu_array, std_array, title='time series prediction',
                save_to=osp.join(path_results, 'test_timeseries_pred.png'))

