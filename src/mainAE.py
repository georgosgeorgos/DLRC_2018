import os
import time
import torch
import argparse
import numpy as np
import pandas as pd
import os.path as osp

from utils.utils import path_exists

from train_routines.trainAEimg import trainAE
from test_routines.testAEimg import testAE

lidar_input_size = 9  # number lidars obs var
joint_input_size = 7  # joint state   cond var
n_samples_y      = 10 # length timeseries
n_samples_z      = 10 # sample from selector
clusters         = 2  # clustering component (background/self | static/dynamic)
split            = "train"
ckpt  = "ckpt_depth.pth"

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--learning_rate", type=float, default=0.0001)
parser.add_argument("--ckpt_dir", type=str, default="../experiments/ckpt/depth/")
parser.add_argument("--data_dir", type=str, default="../DEPTH/")
parser.add_argument("--split", type=str, default=split)
parser.add_argument("--ckpt", type=str, default=ckpt)

args = parser.parse_args()

path_results = osp.join('..', 'experiments', 'AEonDepthImg')
path_exists(path_results)
path_exists(osp.join(path_results, args.ckpt_dir))

if __name__ == '__main__':
	if args.split == "train":
		trainAE(args)
	else:
		testAE(args)
