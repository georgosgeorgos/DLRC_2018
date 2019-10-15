import argparse
import os.path as osp

from test_routines.testCVAE_gmm import test
from train_routines.trainCVAE_gmm import train
from utils.utils import path_exists

epochs = 100
batch_size = 64
batch_size_test = 1
learning_rate = 0.0001
lidar_input_size = 9  # number lidars obs var
joint_input_size = 7  # joint state   cond var
n_samples_y = 10  # length timeseries
n_samples_z = 10  # sample from selector
clusters = 2  # clustering component (background/self | static/dynamic)
split = "train"
ckpt_test = "ckpt_cvae_gmm.pth"

result_dir = osp.join("..", "experiments", "CVAE_gmm")
path_exists(result_dir)

ckpt_dir = osp.join(result_dir, "ckpt/")
path_exists(ckpt_dir)

data_dir = "./data_mockup/"

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=epochs)
parser.add_argument("--batch_size", type=int, default=batch_size)
parser.add_argument("--batch_size_test", type=int, default=batch_size_test)
parser.add_argument("--learning_rate", type=float, default=learning_rate)
parser.add_argument("--lidar_input_size", type=int, default=lidar_input_size)
parser.add_argument("--joint_input_size", type=int, default=joint_input_size)
parser.add_argument("--n_samples_y", type=int, default=n_samples_y)
parser.add_argument("--n_samples_z", type=int, default=n_samples_z)
parser.add_argument("--encoder_layer_sizes", type=list, default=[(lidar_input_size * n_samples_y), 256, 256])
parser.add_argument("--decoder_layer_sizes", type=list, default=[(joint_input_size * n_samples_y), 256, 256])
parser.add_argument("--latent_size", type=int, default=lidar_input_size * clusters)
parser.add_argument("--conditional", action="store_true")
parser.add_argument("--num_labels", type=int, default=0)
parser.add_argument("--ckpt_dir", type=str, default=ckpt_dir)
parser.add_argument("--ckpt", type=str, default=ckpt_test)
parser.add_argument("--split", type=str, default=split)

args = parser.parse_args()

if __name__ == "__main__":
    if args.split == "train":
        train(args)
    else:
        test(args)
