import matplotlib.pyplot as plt
import numpy as np
import torch
import torch as th
from loaders.load_panda_timeseries import PandaDataSetTimeSeries as Dataset
from models.clusteringVAE.model_gmm_selector import VAE as Model
from objectives.loss_gmm_selector import LossSelector as Loss
from torch.utils.data import DataLoader
from utils.utils import move_to_cuda, ckpt_utc, path_exists, tensor_to_variable

def train(args):
    ckpt = ckpt_utc()
    loss_fn = Loss(
              args.batch_size, 
              args.n_samples_y, 
              args.lidar_input_size, 
              args.n_clusters, 
              model_type=args.model_type,
              is_entropy=args.is_entropy,
              lmbda=args.lmbda
              )
    model = Model(
            encoder_layer_sizes=args.encoder_layer_sizes,
            latent_size=args.latent_size,
            n_clusters=args.n_clusters,
            batch_size=args.batch_size,
            model_type=args.model_type,
            )
    model = move_to_cuda(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    optimizer.zero_grad()

    dataset = Dataset(path=args.data_dir, split=args.split, n_samples=args.n_samples_y)
    # randomize an auxiliary index because we want to use random sample of time-series (10 time steps)
    # but the time series have to be intact
    data_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=False)

    dataset_val = Dataset(path=args.data_dir, split="val", n_samples=args.n_samples_y)
    data_loader_val = DataLoader(dataset=dataset_val, batch_size=args.batch_size, shuffle=False)

    loss_train, loss_val = [], []

    for epoch in range(args.epochs):
        model.train()
        dataset.generate_index()
        print("Epoch: ", epoch)
        loss_epoch = []
        for itr, batch in enumerate(data_loader):
            # observable
            y, x = batch
            y = tensor_to_variable(y)
            x = tensor_to_variable(x)
            mu_c, std_c, clusters = model(x)

            loss = loss_fn(y, mu_c, std_c, clusters)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # compute the loss averaging over epochs and dividing by batches
            loss_epoch.append(loss.cpu().data.numpy())

        print("train loss:", np.mean(loss_epoch))
        loss_train.append(np.mean(loss_epoch))

        
        if epoch % args.test_every_n_epochs == 0:
            model.eval()
            loss_epoch = []
            with th.no_grad():
                for itr, batch in enumerate(data_loader):
                    y, x = batch
                    mu_c, std_c, clusters = model(x)
                    loss = loss_fn(y, mu_c, std_c, clusters)
                    loss_epoch.append(loss.cpu().data.numpy())

            print("val loss:", np.mean(loss_epoch))
            loss_val.append(np.mean(loss_epoch))

    plt.plot(np.array(loss_train))
    plt.plot(np.array(loss_val))
    plt.grid()
    plt.show()

    path_exists(args.ckpt_dir)
    th.save(model.state_dict(), args.ckpt_dir + ckpt)
    print("done!")