import time
import argparse
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
import torch as th
from models.cvae_gmm.cvae_gmm import VAE
from loaders.load_panda_timeseries import Loader

def test(args):
    #10 samples y and 57% accuracy
    model = VAE(
            encoder_layer_sizes=args.encoder_layer_sizes,
            decoder_layer_sizes=args.decoder_layer_sizes,
            latent_size=args.latent_size,
            batch_size=args.batch_size_test,
            conditional=args.conditional,
            num_labels=args.num_labels
            )
    
    saved_state_dict = th.load(args.ckpt_dir + args.ckpt)
    model.load_state_dict(saved_state_dict)
    
    dataset = Loader(split=args.split, samples=args.n_samples_y)
    data_loader = DataLoader(dataset=dataset, batch_size=args.batch_size_test, shuffle=False)

    prediction = []

    for epoch in range(args.epochs):
        for itr, y in enumerate(data_loader):
            # observable

            if args.conditional:
                z_y = model.inference(y, x)
            else:
                z_y = model.inference(y)

            z = z_y.data.numpy().squeeze()

            z = np.argmax(z, axis=1)
            prediction.append(z)

    prediction = np.array(prediction)
    np.savetxt("prediction.csv", prediction[:,3].astype(int), fmt='%i')
    print("Done!")