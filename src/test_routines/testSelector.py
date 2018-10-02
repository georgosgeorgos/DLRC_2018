import time
import argparse
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
import torch as th
from src.loaders.load_panda_timeseries import Loader
from src.models.cvae_gmm.cvae_gmm_selector import VAE
from src.objectives.loss_selector import LossSelector
import matplotlib.pyplot as plt


def test(args):
    model = VAE(
            encoder_layer_sizes=args.encoder_layer_sizes,
            latent_size=args.latent_size,
            n_clusters=args.n_clusters,
            batch_size=args.batch_size_test
            )
    
    saved_state_dict = th.load(args.ckpt_dir + args.ckpt)
    model.load_state_dict(saved_state_dict)
    
    dataset = Loader(path=args.data_dir, split="val", samples=args.n_samples_y)
    print(args.data_dir)
    print(args.split)
    data_loader = DataLoader(dataset=dataset, batch_size=args.batch_size_test, shuffle=False)
    print(model)

    res = {"y": [], "mu": [], "std": []}
    prediction = []
    model.eval()
    dataset.generate_index()
    for itr, batch in enumerate(data_loader):
        y, x = batch
        
        y = y.view(-1, args.n_samples_y, args.lidar_input_size)
        y = y[:,-1,:]

        mu_c, std_c, clusters = model(x)
        # observable
        prediction.append((y.data.numpy() - mu_c.data.numpy()) / std_c.data.numpy())
        res["y"].append(y.data.numpy().tolist())
        res["mu"].append(mu_c.data.numpy().tolist())
        res["std"].append(std_c.data.numpy().tolist())

    prediction = np.array(prediction).squeeze()
    import json
    with open("res.json", "w") as f:
        json.dump(res, f)

    import seaborn as sns
    f, ax = plt.subplots(prediction.shape[1], 1, figsize=(20,20))
    for i in range(prediction.shape[1]):
        sns.distplot(prediction[:,i], kde=True, ax=ax[i])
        ax[i].set_xlim([-3, 3])
    plt.show()
    np.savetxt("prediction.csv", prediction[:,3].astype(int), fmt='%i')
    print("Done!")