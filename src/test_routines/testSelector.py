import time
import argparse
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
import torch as th
rom loaders.load_panda_timeseries import PandaDataSetTimeSeries as Dataset
from models.clusteringVAE.model_gmm_selector import VAE as Model
import matplotlib.pyplot as plt
from utils.utils import plot_timeseries
import os.path as osp
import json

def test(args):
    model = Model(
            encoder_layer_sizes=args.encoder_layer_sizes,
            latent_size=args.latent_size,
            n_clusters=args.n_clusters,
            batch_size=args.batch_size_test,
            flag=args.model_type,
            )
    
    saved_state_dict = th.load(args.ckpt_dir + args.ckpt)
    model.load_state_dict(saved_state_dict)

    dataset = Dataset(path=args.data_dir, split=args.flag, n_samples=args.n_samples_y)
    data_loader = DataLoader(dataset=dataset, batch_size=args.batch_size_test, shuffle=False)

    res = {"y": [], "mu": [], "std": [], "cluster": []}
    prediction = []
    model.eval()
    
    dataset.generate_index()
    for itr, batch in enumerate(data_loader):
        y, x = batch

        y = y.view(-1, args.n_samples_y, args.lidar_input_size)
        y = y[:,-1,:]

        mu_c, std_c, clusters = model(x)
        # observable
        res["y"].append(y.data.numpy().squeeze().tolist())
        res["mu"].append(mu_c.data.numpy().squeeze().tolist())
        res["std"].append(std_c.data.numpy().squeeze().tolist())
        res["cluster"].append(clusters.data.numpy().squeeze().tolist())
    
    with open(osp.join(args.result_dir, "res_" + args.flag + ".json"), "w") as f:
        json.dump(res, f)

    mu_array    = np.array(res["mu"])
    std_array   = np.array(res["std"])
    input_array = np.array(res["y"])
    plot_timeseries(input=input_array, pred=mu_array, std=std_array, xlabel="time", ylabel="depth (m)",
                title='time series prediction', save_to=osp.join(args.result_dir, 'timeseries_pred_selector_' + args.flag + '.png'))
    print("Done!")