import datetime
from os import makedirs
from os.path import exists

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from matplotlib import gridspec
from torch.autograd import Variable

sns.set(style="darkgrid")
sns.axes_style(style={"axes.grid": True})


def tensor_to_variable(x, required_grad=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=required_grad)

def std(log_var):
    res = torch.exp(0.5 * log_var)
    return res

def reshape(z, l=9, s=2):
    res = torch.zeros((z.size()[0], l, s))
    index = 0
    for j in range(s):
        res[:, :, j] = z[:,(index*l):((index+1)*l)]
        index +=1
    return res

def expand(x, s=2):
    n, m = x.size()
    x_expanded = torch.zeros(n, m, 2)
    for j in range(s):
        x_expanded[:, :, j] = x
    return x_expanded


def path_exists(path):
    if not exists(path):
        makedirs(path)
    return path

def ckpt_utc():
    s = datetime.datetime.utcnow()
    s = str(s).split(".")[0]
    s = s.split(" ")
    s = "_".join(s)
    ckpt = "ckpt_" + s + ".pth"
    return ckpt


def move_to_cuda(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return x


def plot_eval(x=None, y=None, xlabel=None, ylabel=None, title=None, save_to=None):
    plt.figure(figsize=(7, 5))
    plt.plot(x, y)
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title(title)
    plt.savefig(save_to, format='png')
    plt.tight_layout()
    plt.close()


def plot_hist(x, title, save_to):

    num_samples, num_channels = x.shape
    nrow, ncol = 3, 3
    colors = cm.viridis(np.linspace(0, 1, num_channels))
    fig = plt.figure(figsize=(15*nrow, 10*ncol))
    fig.suptitle(title, size=40)

    for idx in range(num_channels):
        row = int(idx % nrow)
        col = int(idx / ncol)
        gs = gridspec.GridSpec(nrow, ncol, width_ratios=[1]*nrow, height_ratios=[1]*ncol)
        ax0 = plt.subplot(gs[row, col])
        sns_dist = sns.distplot(x[:, idx], ax=ax0, vertical=False, bins=20, kde=True,
                     hist_kws={"color": colors[idx], "range": (-4, 4)},
                     kde_kws={"color": colors[idx], "lw": 3}, label='channel {}'.format(idx))
        ax0.legend(loc=1, fontsize=30)
    plt.savefig(save_to, format='png')
    plt.close()