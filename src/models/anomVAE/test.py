import glob
import os.path as osp

import torch as th
from torch.utils.data import DataLoader
from torchvision import transforms

from src.loaders.load_panda import PandaDataSet
from src.utils import configs as cfg
from src.utils.utils import path_exists, ckpt_utc, plot_eval
from src.models.anomVAE.anomVAE import VAE
from torch.nn import functional as F
import numpy as np
import os

tebs = 100
th.manual_seed(42)
cuda = th.cuda.is_available()
device = th.device("cuda" if cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
rootdir = '../../../'
path_results = osp.join(rootdir, 'experiments', 'anomVAE')
path_exists(path_results)
ckpt = ckpt_utc()
ckpt_dir = osp.join(path_results, 'ckpt/')
path_exists(ckpt_dir)

test_set = PandaDataSet(root_dir=osp.join(rootdir, 'data'), filename='anomaly_detection_gt.pkl', train=True,
                        test_split=0.0, transform=transforms.Compose([
        transforms.Lambda(lambda n: th.Tensor(n)),
        transforms.Lambda(
            lambda n: th.Tensor.clamp(n, cfg.LIDAR_MIN_RANGE, cfg.LIDAR_MAX_RANGE)),
        transforms.Lambda(lambda n: n / 1000)
    ]))
test_loader = DataLoader(test_set, batch_size=tebs, shuffle=False, **kwargs)

model = VAE().to(device)

saved_state_dict = th.load(glob.glob(osp.join(ckpt_dir, "*.pth"))[0])
model.load_state_dict(saved_state_dict)

depth_lidars = []
recon_input_error = []

for i, (x, y, _) in enumerate(test_loader):
    bs, _, = y.size()

    x = x.to(device).float()
    y = y.to(device).float()

    input_concat = th.cat((x, y), dim=1)

    latent, input_recon, mu, logvar = model(input_concat)

    l2_dist = th.norm((input_concat - input_recon), p=2, dim=1)

    recon_input_error.append((l2_dist.cpu().data.numpy().tolist()))
    depth_lidars.append(y.cpu().data.numpy().tolist())

depth_lidars_array = np.array(depth_lidars)
depth_lidars_array = depth_lidars_array.reshape((depth_lidars_array.shape[0] * depth_lidars_array.shape[1], 9))

recon_input_error_array = np.array(recon_input_error)
recon_input_error_array = recon_input_error_array.reshape(
    (recon_input_error_array.shape[0] * recon_input_error_array.shape[1]))

plot_eval(x=np.arange(len(depth_lidars_array)), y=depth_lidars_array[:, 3], xlabel='timesteps', ylabel='depth (m)',
          title='input anomaly gt', save_to=os.path.join(path_results, 'input_anom_gt.png'))

plot_eval(x=np.arange(len(recon_input_error_array)), y=recon_input_error_array, xlabel='timesteps',
          ylabel='l2 distance (input and recon)',
          title='input anomaly gt', save_to=os.path.join(path_results, 'input_recon_l2_dist.png'))
