import numpy as np
import torch as th
import glob
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data import DataLoader
import torch as th
from visualization.normalMLP import  NormalMLP
import os.path as osp
from torch.distributions.normal import Normal
import pickle as pkl 

from src.models.clusteringVAE.model_gmm_selector import VAE as ModelCluster

pf = "./robot_sampling/data_0.pkl"
pf = "../data/train_data_correct.pkl"

import pickle as pkl
from random import shuffle
from torchvision import transforms
import numpy as np
import torch as th
from torch.utils import data
import src.utils.utils as cfg


class PandaDataSetTimeSeries(data.Dataset):


    def routine(self, x_data, i):
        x = x_data[(i - self.n_samples):i]
        x = x.flatten()
        x = th.from_numpy(x).float()
        return x

    def __getitem__(self, index):
        if self.split not in ["test"]:
            i = self.index_lidar[index]
        else:
            i = index

        if i < self.n_samples:
            i += self.n_samples

        Y = self.data_lidar[(i - self.n_samples):i]
        Y = th.from_numpy(Y).float()

        X = self.routine(self.data_joint, i)

class Sampler:
    def __init__(self, n=100):
        with open(pf, "rb") as f:
            self.data = pkl.load(f)

            self.runs = len(self.data)
            self.data_lidar   = []
            self.data_joint   = []
            self.data_joint_v = []
            for i in range(self.runs):
                self.data_lidar.extend(self.data[i]["lidar"]["measurements"])
                self.data_joint.extend(self.data[i]["state"]["j_pos"])
                self.data_joint_v.extend(self.data[i]["state"]["j_vel"])
            
            self.data_lidar = np.array(self.data_lidar, dtype=float)
            self.data_lidar /= 1000
            self.data_lidar[self.data_lidar > 2.0] = 2.0

            self.data_joint = np.array(self.data_joint, dtype=float)
            self.data_joint_v = np.array(self.data_joint_v, dtype=float)
        self.n = n

    def get_sample_anomaly(self, n_interval):
        n_interval = n_interval % self.data_lidar.shape[0]
        sample_lidar   = self.data_lidar[(n_interval*self.n):((n_interval*self.n)+self.n)]
        sample_joint   = self.data_joint[(n_interval*self.n):((n_interval*self.n)+self.n)]
        sample_joint_v = self.data_joint_v[(n_interval*self.n):((n_interval*self.n)+self.n)]
        return (sample_lidar, sample_joint, sample_joint_v)

    def get_sample_clustering(self, n_interval):
        return None

class Probs:
    def __init__(self, n=100, l=3):
        self.cuda = th.cuda.is_available()
        self.device = th.device("cuda" if self.cuda else "cpu")
        self.n = n
        self.l = l

        self.sampler = Sampler(self.n)

        self.modelAnomalyDetection = NormalMLP().to(self.device)
        self.modelAnomalyDetection.load_state_dict(th.load("../experiments/normalMLP/ckpt/ckpt.pkl", map_location=self.device))
        self.modelAnomalyDetection.eval()

        self.n_clusters = 2
        self.lidar_input_size = 9
        self.joint_input_size = 7
        self.n_samples_y = 10
        self.latent_size=self.lidar_input_size * self.n_clusters
        self.encoder_layer_sizes = [self.joint_input_size*self.n_samples_y, 256, 256]

        self.modelCluster = ModelCluster(
            encoder_layer_sizes=self.encoder_layer_sizes,
            latent_size=self.latent_size,
            n_clusters=self.n_clusters,
            batch_size=self.n,
            model_type="selector",
        )

        self.modelCluster.load_state_dict(th.load("../experiments/selector_no_entropy/ckpt/ckpt_selector.pth", map_location=self.device))
        self.modelCluster.eval().cpu()

    def get_data(self, n_interval):
        y, x, _ = self.sampler.get_sample_anomaly(n_interval)

        x = th.from_numpy(x).to(self.device).float()
        y = th.from_numpy(y).to(self.device).float()

        mu, logvar = self.modelAnomalyDetection(x)
        std  = th.exp(0.5 * logvar)
        N    = Normal(mu, std)
        z = np.abs((y - mu) / std)
        z > 4 # outlier
        # four or five sigma
        # too low or too high two side confidence interval do with z variable
        prob = 1 - N.cdf(y) 

        #_, _, clusters = self.modelCluster(x[:10,:])

        y    = y.cpu().data.numpy()[:,self.l]
        mu   = mu.cpu().data.numpy()[:,self.l]
        std  = std.cpu().data.numpy()[:,self.l]
        prob = prob.cpu().data.numpy()[:,self.l]
        #cluster = clusters.data.numpy().squeeze()[:,self.l]

        data =  {"input": y, "mu": mu, "std": std, "prob": prob} #"cluster": cluster}
        return data

if __name__ == '__main__':
    p =Probs()
    d =p.get_data(3)
    print(d)