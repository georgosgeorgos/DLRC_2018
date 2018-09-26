from os import makedirs
from os.path import exists
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
import numpy as np
from matplotlib import gridspec
sns.set(style="darkgrid")
sns.axes_style(style={"axes.grid": True})


def path_exists(path):
    if not exists(path):
        makedirs(path)
    return path


def plot_eval(x, y, title, save_to):

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
    plt.figure(figsize=(15*nrow, 10*ncol))
    plt.title(title)

    for idx in range(num_channels):
        row = int(idx % nrow)
        col = int(idx / ncol)
        gs = gridspec.GridSpec(nrow, ncol, width_ratios=[1]*nrow, height_ratios=[1]*ncol)
        ax0 = plt.subplot(gs[row, col])

        sns.distplot(x[:, idx], ax=ax0, vertical=False, bins=15, kde=True,
                     hist_kws={"color": colors[idx], "range": (np.min(x), np.max(x))},
                     kde_kws={"color": colors[idx], "lw": 3}, label='channel {}'.format(idx))
        ax0.legend(loc=1, fontsize=30)
    plt.tight_layout()
    plt.savefig(save_to, format='png')
    plt.close()
