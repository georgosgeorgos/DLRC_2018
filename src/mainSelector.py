import argparse
import os.path as osp
from utils.utils import path_exists
from test_routines.testSelector import test
from train_routines.trainSelector import train

epochs           = 100
batch_size       = 64
batch_size_test  = 1
learning_rate    = 0.0001
lidar_input_size = 9       # number lidars obs var
joint_input_size = 7       # joint state   cond var
n_samples_y      = 1      # length timeseries
n_samples_z      = 10      # sample from selector
n_clusters       = 2       # clustering component (background/self | static/dynamic)
split            = "train"
test_every_n_epochs = 10
split_evaluation    = "val"
model_type          = "selector"
ckpt_test  = "ckpt_" + model_type + ".pth"
is_entropy          = False
is_multimodal       = True
lmbda               = 1
variant             = "1_sample"

if is_entropy:
	result_dir = osp.join('..', 'experiments', model_type + str(lmbda) + variant)
else:
	result_dir = osp.join('..', 'experiments', model_type + "_no_entropy" + variant)
path_exists(result_dir)

ckpt_dir = osp.join(result_dir, 'ckpt/')
path_exists(ckpt_dir)

data_dir = "../data/"

parser = argparse.ArgumentParser()
parser.add_argument("--epochs",              type=int,   default=epochs)
parser.add_argument("--batch_size",          type=int,   default=batch_size)
parser.add_argument("--batch_size_test",     type=int,   default=batch_size_test)
parser.add_argument("--learning_rate",       type=float, default=learning_rate)
parser.add_argument("--lidar_input_size",    type=int,   default=lidar_input_size)
parser.add_argument("--joint_input_size",    type=int,   default=joint_input_size)
parser.add_argument("--n_samples_y",         type=int,   default=n_samples_y)
parser.add_argument("--n_samples_z",         type=int,   default=n_samples_z)
parser.add_argument("--encoder_layer_sizes", type=list,  default=[(joint_input_size*n_samples_y), 256, 256])
parser.add_argument("--n_clusters",          type=int,   default=n_clusters)
parser.add_argument("--latent_size",         type=int,   default=lidar_input_size*n_clusters)
parser.add_argument("--conditional",         action='store_true')
parser.add_argument("--num_labels",          type=int,   default=0)
parser.add_argument("--ckpt_dir",            type=str,   default=ckpt_dir)
parser.add_argument("--data_dir",            type=str,   default=data_dir)
parser.add_argument("--result_dir",          type=str,   default=result_dir)
parser.add_argument("--ckpt_test",           type=str,   default=ckpt_test)
parser.add_argument("--split",               type=str,   default=split)
parser.add_argument("--test_every_n_epochs", type=str,   default=test_every_n_epochs)
parser.add_argument("--split_evaluation",    type=str,   default=split_evaluation)
parser.add_argument("--model_type",          type=str,   default=model_type)
parser.add_argument("--is_entropy",          type=str,   default=is_entropy)
parser.add_argument("--is_multimodal",          type=str,   default=is_multimodal)
parser.add_argument("--lmbda",               type=int,   default=lmbda)

args = parser.parse_args()

if __name__ == '__main__':
	if args.split == "train":
		train(args)
	else:
		test(args)
