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

sns.set(style="whitegrid")
sns.axes_style(style={"axes.grid": True})


def tensor_to_variable(x, required_grad=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=required_grad)

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


def cumulative_moving_average(x=None, x_new=None, n=None):
    """
    # Cumulative moving average https://en.wikipedia.org/wiki/Moving_average
    :param x: float, current average
    :param x_new: float, new value to update average on
    :param n: int, iteration
    :return:
    """
    return (x_new + n * x) / (n + 1)


def plot_eval(x=None, y=None, xlabel=None, ylabel=None, title=None, save_to=None):
    plt.clf()
    plt.figure(figsize=(7, 5))
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(save_to, format='png')
    plt.tight_layout()
    plt.close()


def plot_scatter(x=None, y=None, xlabel=None, ylabel=None, title=None, save_to=None):
    plt.clf()
    plt.figure(figsize=(10, 10))
    plt.scatter(x, y, s=15)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(save_to, format='png')
    plt.tight_layout()
    plt.close()


def plot_hist_lidars(x, title, save_to):
    plt.clf()
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


def plot_hist(x, xlabel=None, ylabel=None, title=None, save_to=None):
    plt.clf()
    fig = plt.figure(figsize=(10, 5))

    sns_dist = sns.distplot(x, vertical=False, bins=25, kde=False,
                 hist_kws={"color": "blue", "range": (-4, 4)},
                 kde_kws={"color": "blue", "lw": 3})
    # # N(0,1)
    # sns_dist = sns.distplot(np.random.randn(len(x)), vertical=False, bins=25,
    #                         hist_kws={"color": "red", "range": (-3, 3)},
    #                         kde_kws={"color": "red", "lw": 3})
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(save_to, format='png')
    plt.close()


def plot_correlation_matrix(x, xlabel=None, ylabel=None, title=None, save_to=None):
    plt.clf()
    fig = plt.figure(figsize=(9, 7))
    cor = np.corrcoef(x.T)
    sns.heatmap(cor, center=0., vmin=-1., vmax=1.)
    plt.title(title)
    plt.savefig(save_to, format='png')
    plt.close()


def plot_timeseries(input, pred, mu, std, num_std=3, downsample_step=1, xlabel=None, ylabel=None, title=None, save_to=None):
    plt.clf()
    num_samples, num_channels = input.shape
    colors = cm.viridis(np.linspace(0, 1, num_channels))
    downsample_every_nth = [i for i in range(0, input.shape[0], downsample_step)]

    fig = plt.figure(figsize=(40, 60))

    for idx in range(num_channels):
        x = idx % num_channels
        y = idx % 1
        gs = gridspec.GridSpec(num_channels, 2, width_ratios=[4, 1], height_ratios=[1]*num_channels)
        ax0 = plt.subplot(gs[x, y])
        plt.xlabel('time')
        plt.ylabel('distance')
        ax0.plot(input[downsample_every_nth, idx], color='black')
        #ax0.plot(pred[downsample_every_nth, idx], color='green')
        ax0.plot(mu[downsample_every_nth, idx], color='red')
        ax0.plot(mu[downsample_every_nth, idx] - std[downsample_every_nth, idx] * num_std, color='blue', alpha=0.5)
        ax0.plot(mu[downsample_every_nth, idx] + std[downsample_every_nth, idx] * num_std, color='blue', alpha=0.5)

        #ax0.fill_between([i for i in range(input[downsample_every_nth].shape[0])], mu[downsample_every_nth, idx] - std[downsample_every_nth, idx] * num_std, facecolor="blue", alpha=0.1)
        #ax0.fill_between([i for i in range(input[downsample_every_nth].shape[0])], mu[downsample_every_nth, idx] + std[downsample_every_nth, idx] * num_std, facecolor="blue", alpha=0.1)

        ax1 = plt.subplot(gs[x, y + 1])
        vert_hist = np.histogram(input[downsample_every_nth, idx])
        sns.distplot(input[downsample_every_nth, idx], ax=ax1, vertical=False, bins=15, kde=False,
                     hist_kws={"color": "blue", "range": (np.min(input), np.max(input))},
                     kde_kws={"color": "blue", "lw": 3})
        plt.xlabel('frequency')
    plt.tight_layout()
    plt.title(title)
    plt.savefig(save_to, format='png')
    plt.close()
