import sys
sys.path.append('/home/georgos/DLRC_2018/')

import numpy as np
import torch as th
import glob
import src
import visualization
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
import pickle as pkl
from random import shuffle
from torchvision import transforms
import numpy as np
import torch as th
from torch.utils import data
import src.utils.utils as cfg


pf = "./robot_sampling/data_0.pkl"
pf = "../data/train_data_correct.pkl"

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

    def get_sample_clustering(self, n_interval, n_sample_y=10):
        sample_lidar_clustering   = []
        sample_joint_clustering   = []
        sample_joint_v_clustering = []
        start = n_interval*self.n
        end   = start + self.n
        if start < n_sample_y:
            start = n_sample_y
            end = end * n_sample_y

        for sample in range(start, end, 1):
            sample_lidar_clustering.append(self.data_lidar[(sample-n_sample_y+1):(sample+1)])
            sample_joint_clustering.append(self.data_joint[(sample-n_sample_y+1):(sample+1)].flatten())
            sample_joint_v_clustering.append(self.data_joint_v[(sample-n_sample_y+1):(sample+1)].flatten())

        sample_lidar_clustering   = np.array(sample_lidar_clustering)
        sample_joint_clustering   = np.array(sample_joint_clustering)
        sample_joint_v_clustering = np.array(sample_joint_v_clustering)
        return (sample_lidar_clustering, sample_joint_clustering, sample_joint_v_clustering)


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
        self.test = "cdf"

        self.modelCluster = ModelCluster(
            encoder_layer_sizes=self.encoder_layer_sizes,
            latent_size=self.latent_size,
            n_clusters=self.n_clusters,
            batch_size=self.n,
            model_type="selector",
        )

        self.modelCluster.load_state_dict(th.load("../experiments/selector_no_entropy/ckpt/ckpt_selector.pth", map_location=self.device))
        self.modelCluster.eval().cpu()

    def routine_tensor(self, x):
        x = th.from_numpy(x).to(self.device).float()
        return x

    def routine_array(self, x):
        x = x.cpu().data.numpy()
        return x

    def outlier_test(self, y, mu, std):
        N = Normal(mu, std)
        if self.test == "z":
            z = np.abs((y - mu) / std)
            outlier = z[z > 3]
        else:
            prob = 1 - N.cdf(y)
        return prob

    def get_data(self, n_interval):
        y_an, x_an, _ = self.sampler.get_sample_anomaly(n_interval)
        x_an = self.routine_tensor(x_an)
        y_an = self.routine_tensor(y_an)

        y_cl, x_cl, _ = self.sampler.get_sample_clustering(n_interval)
        x_cl = self.routine_tensor(x_cl)
        y_cl = self.routine_tensor(y_cl)
        
        mu_an, logvar_an = self.modelAnomalyDetection(x_an)
        std_an  = th.exp(0.5 * logvar_an)

        prob_an = self.outlier_test(y_an, mu_an, std_an)

        print(x_an.size(),y_an.size(),x_cl.size(),y_cl.size()) 

        _, _, clst = self.modelCluster(x_cl)
        #print(y_an[-1])
        #print(y_cl[-1][-1])

        y_an    = self.routine_array(y_an)[:,self.l]
        mu_an   = self.routine_array(mu_an)[:,self.l]
        std_an  = self.routine_array(std_an)[:,self.l]
        prob_an = self.routine_array(prob_an)[:,self.l]
        
        clst    = self.routine_array(clst)[:,self.l]

        data =  {"input": y_an, "mu": mu_an, "std": std_an, "prob": prob_an, "cluster": clst}
        return data

if __name__ == '__main__':
    p =Probs(n=100)
    d =p.get_data(3)
    print(d["cluster"])
    print(d["cluster"].argmax(axis=1))

    for key in d:
        print(d[key].shape)