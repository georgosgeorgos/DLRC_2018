import time
import argparse
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
import torch as th
from loaders.load_panda_timeseries import PandaDataSetTimeSeries as Dataset
from models.clusteringVAE.model_gmm_selector import Encoder as Model
import matplotlib.pyplot as plt
from utils.utils import plot_timeseries
import os.path as osp
import json

def test(args):
    model = Model(
            layer_sizes=args.encoder_layer_sizes,
            latent_size=args.latent_size,
            is_uq=False
            )
    
    saved_state_dict = th.load(args.ckpt_dir + args.ckpt_test)
    model.load_state_dict(saved_state_dict)

    dataset = Dataset(path=args.data_dir, 
                      split=args.split_evaluation, 
                      n_samples=args.n_samples_y, 
                      is_label_y=True)

    data_loader = DataLoader(dataset=dataset, batch_size=args.batch_size_test, shuffle=False)

    res = {"y": [], "mu": [], "std": [], "cluster": []}
    model.eval()
    
    dataset.generate_index()
    acc = []
    lbl_list  = []
    pred_list = []
    for itr, batch in enumerate(data_loader):
        y, x, lbl = batch

        state = th.cat([y, x], dim=1)
        pred = model(state)
        pred = pred.reshape(-1, args.n_clusters, args.lidar_input_size).permute(0,2,1)

        y = y.view(-1, args.n_samples_y, args.lidar_input_size)
        y = y[:,-1,:]

        pred = pred.argmax(dim=2)
        lbl  = lbl.argmax(dim=2)
        
        lbl_list.append(lbl.data.numpy()[0])
        pred_list.append(pred.data.numpy()[0])

        if args.model_type == "gmm":
            index = clusters.argmax(dim=2).squeeze()
            mu_new = th.zeros((1, index.size()[0]))
            std_new = th.zeros((1, index.size()[0]))

            for i in range(len(index)):
                mu_new[:,i]  = mu_c[:,i,index[i]]
                std_new[:,i] = std_c[:,i,index[i]]

            mu_c  = mu_new
            std_c = std_new


    lbl_array = np.array(lbl_list)
    pred_array = np.array(pred_list)
    np.savetxt("lbl.npy", lbl_list)
    np.savetxt("pred.npy", pred_list)

    print((lbl_array == pred_array).mean())
        # observable
        #res["y"].append(y.data.numpy().squeeze().tolist())
        #res["mu"].append(mu_c.data.numpy().squeeze().tolist())
        #res["std"].append(std_c.data.numpy().squeeze().tolist())
        #res["cluster"].append(clusters.data.numpy().squeeze().tolist())
    
    #with open(osp.join(args.result_dir, "res_" + args.split_evaluation + ".json"), "w") as f:
    #    json.dump(res, f)

    #mu_array    = np.array(res["mu"])
    #std_array   = np.array(res["std"])
    #input_array = np.array(res["y"])
    #plot_timeseries(input=input_array, pred=mu_array, std=std_array, 
    #                xlabel="time", ylabel="depth (m)", title='time series prediction', 
    #               save_to=osp.join(args.result_dir, 'timeseries_pred_' + args.split_evaluation + '.png'))
    print("Done!")