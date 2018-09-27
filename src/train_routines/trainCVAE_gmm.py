import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

import torch as th

from models.cvae_gmm.cvae_gmm import VAE
from objectives.nELBO_gmm import nELBO
from loaders.load_panda_timeseries import Loader
from utils.utils import move_to_cuda, ckpt_utc, path_exists, tensor_to_variable
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch as th
from loaders.load_panda_timeseries import Loader
from models.cvae_gmm.cvae_gmm import VAE
from objectives.nELBO_gmm import nELBO
from torch.utils.data import DataLoader
from utils.utils import move_to_cuda, ckpt_utc, path_exists, tensor_to_variable


def train(args):

    split = args.split
    ckpt = ckpt_utc()
    
    loss_fn = nELBO(args.batch_size, args.n_samples_z, args.n_samples_y)

    model = VAE(
            encoder_layer_sizes=args.encoder_layer_sizes,
            decoder_layer_sizes=args.decoder_layer_sizes,
            latent_size=args.latent_size,
            batch_size=args.batch_size,
            conditional=args.conditional,
            num_labels=args.num_labels
            )
    model = move_to_cuda(model)
    model.train()

    # probably adam not the most appropriate algorithms
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    optimizer.zero_grad()

    dataset = Loader(split=args.split, samples=args.n_samples_y)
    # randomize an auxiliary index because we want to use sample of time-series (10 time steps)
    data_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=False)

    loss_list = []
    for epoch in range(args.epochs):
        dataset.generate_index()
        print("Epoch: ", epoch)
        L = []
        for itr, batch in enumerate(data_loader):
            # observable
            y, x = batch
            y = tensor_to_variable(y)
            x = tensor_to_variable(x)
            if y.size(0) != args.batch_size:
                continue
            else:
                mu_phi, log_var_phi, mu_theta, log_var_theta = model(y, x)

                loss, kld, ll = loss_fn(y, mu_phi, log_var_phi, mu_theta, log_var_theta)

                if split == 'train':
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                # compute the loss averaging over epochs and dividing by batches
                L.append(loss.cpu().data.numpy())

        print("negative likelihood: ", -ll.cpu().data.numpy())
        print("kl: ", kld.cpu().data.numpy())
        print("loss:", loss.cpu().data.numpy())
        
        loss_list.append(np.mean(L) / (len(data_loader)))

    plt.plot(np.array(loss_list))
    plt.grid()
    plt.show()

    path_exists(args.ckpt_dir)
    th.save(model.state_dict(), args.ckpt_dir + ckpt)
    print("done!")