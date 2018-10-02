import numpy as np
import torch as th
import glob
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data import DataLoader
import torch as th
from src.loaders.load_panda import PandaDataSet
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

test_set = PandaDataSet(root_dir=osp.join(rootdir, 'data'), filename='demos.pkl', train=True, test_split=0.0, transform=transforms.Compose([
    transforms.Lambda(lambda n: th.Tensor(n)),
    transforms.Lambda(
        lambda n: th.Tensor.clamp(n, cfg.LIDAR_MIN_RANGE, cfg.LIDAR_MAX_RANGE)),
    transforms.Lambda(lambda n: n / 1000)
]))
test_loader = DataLoader(test_set, batch_size=tebs, shuffle=False, **kwargs)

model = NormalMLP().to(device)

saved_state_dict = th.load(glob.glob(osp.join(ckpt_dir, "*.pkl"))[0])
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

print(len(input_list), len(input_list[0]), input_list)

input_array = np.array(input_list).squeeze()
pred_array = np.array(pred_list).squeeze()
mu_array = np.array(mu_list).squeeze()
std_array = np.array(std_list).squeeze()

# ## Simple anomaly test
#
# # Inject anomaly for lidar 3 (indexed-0) measurements
# # between timesteps 200-400 --> 0
# input_array[200:401, 3] = 0.
# # between timesteps 600-800 --> 2
# input_array[600:801, 3] = 2.
#
# # Inject anomaly for lidar 0 (indexed-0) measurements
# # between timesteps 200-400 --> 0
# input_array[200:401, 0] = 1.
# # between timesteps 600-800 --> 2
# input_array[600:801, 0] = 1.
#
# # Inject anomaly for lidar 1 (indexed-0) measurements
# # between timesteps 200-400 --> 0
# input_array[1000:, 0] = 0

print(input_array.shape)


plot_timeseries(input=input_array, pred=mu_array, std=std_array, xlabel="time", ylabel="depth (m)",
                title='time series prediction', save_to=osp.join(path_results, 'test_timeseries_pred_anom_real.png'))

