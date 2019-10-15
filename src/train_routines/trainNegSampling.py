import matplotlib.pyplot as plt
import numpy as np
import torch as th
from torch import nn, optim
from loaders.load_panda_timeseries import PandaDataSetTimeSeries as Dataset
from models.clusteringVAE.model_gmm_selector import Encoder as Model
from objectives.loss_neg_sampling import lossNegSampling as Loss
from torch.utils.data import DataLoader
from utils.utils import move_to_cuda, ckpt_utc, path_exists, tensor_to_variable


def train(args):
    ckpt = ckpt_utc()
    loss_fn = Loss()
    model = Model(layer_sizes=args.encoder_layer_sizes, latent_size=args.latent_size, is_uq=False)
    model = move_to_cuda(model)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    optimizer.zero_grad()

    # randomize an auxiliary index because we want to use random sample of time-series (10 time steps)
    # but the time series have to be intact
    dataset = Dataset(
        path=args.data_dir,
        path_images=args.data_dir + "TRAIN_DATA/DEPTH/",
        split=args.split,
        n_samples=args.n_samples_y,
        is_label_y=args.is_label_y,
        is_multimodal=args.is_multimodal,
    )
    data_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=False)

    dataset_val = Dataset(
        path=args.data_dir,
        path_images=args.data_dir + "TRAIN_DATA/DEPTH/",
        split="val",
        n_samples=args.n_samples_y,
        is_label_y=args.is_label_y,
        is_multimodal=args.is_multimodal,
    )
    data_loader_val = DataLoader(dataset=dataset_val, batch_size=args.batch_size, shuffle=False)

    loss_train, loss_val = [], []

    for epoch in range(args.epochs):
        model.train()
        dataset.generate_index()
        print("Epoch: ", epoch)
        loss_epoch = []
        for itr, batch in enumerate(data_loader):
            # observable
            y, x, lbl = batch
            y = tensor_to_variable(y)
            x = tensor_to_variable(x)
            lbl = tensor_to_variable(lbl)
            state = th.cat([y, x], dim=1)

            pred = model(state)
            pred = pred.reshape(-1, args.n_clusters, args.lidar_input_size)  # .permute(0,2,1)

            loss = loss_fn(pred, lbl)
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
                    y, x, lbl = batch
                    state = th.cat([y, x], dim=1)

                    pred = model(state)
                    pred = pred.reshape(-1, args.n_clusters, args.lidar_input_size)  # .permute(0,2,1)

                    loss = loss_fn(pred, lbl)
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
