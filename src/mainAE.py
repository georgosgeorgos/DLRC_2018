import argparse
import os.path as osp

from test_routines.testAEimg import test
from train_routines.trainAEimg import train
from utils.utils import path_exists

lidar_input_size = 9  # number lidars obs var
joint_input_size = 7  # joint state   cond var
n_samples_y = 10  # length timeseries
n_samples_z = 10  # sample from selector
clusters = 2  # clustering component (background/self | static/dynamic)
split = "train"
ckpt_test = "ckpt_depth.pth"

result_dir = osp.join("..", "experiments", "AEImg")
path_exists(result_dir)

ckpt_dir = osp.join(result_dir, "ckpt/")
path_exists(ckpt_dir)

data_dir = "../TRAIN_DATA/DEPTH/"

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--batch_size_test", type=int, default=1)
parser.add_argument("--learning_rate", type=float, default=0.0001)
parser.add_argument("--split", type=str, default=split)
parser.add_argument("--ckpt_test", type=str, default=ckpt_test)
parser.add_argument("--ckpt_dir", type=str, default=ckpt_dir)
parser.add_argument("--data_dir", type=str, default=data_dir)
parser.add_argument("--result_dir", type=str, default=result_dir)

args = parser.parse_args()

if __name__ == "__main__":
    if args.split == "train":
        train(args)
    else:
        test(args)
