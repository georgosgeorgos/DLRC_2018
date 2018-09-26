from os import makedirs, listdir
from os.path import exists
import matplotlib.pyplot as plt


def path_exists(path):
    if not exists(path):
        makedirs(path)
    return path


def plot_eval(x, y, title, savepathfile):
    plt.plot(x, y)

    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title(title)
    plt.savefig(savepathfile, format='png')
    plt.close()