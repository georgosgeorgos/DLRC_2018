import argparse

from test_routines.testCVAE_gmm import test
from train_routines.trainCVAE_gmm import train

lidar_input_size = 9  # number lidars obs var
joint_input_size = 7  # joint state   cond var
n_samples_y      = 10 # length timeseries
n_samples_z      = 10 # sample from selector
clusters         = 2  # clustering component (background/self | static/dynamic)

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--batch_size_test", type=int, default=1)
parser.add_argument("--learning_rate", type=float, default=0.0001)
parser.add_argument("--lidar_input_size", type=int, default=lidar_input_size)
parser.add_argument("--joint_input_size", type=int, default=joint_input_size)
parser.add_argument("--n_samples_y", type=int, default=n_samples_y)
parser.add_argument("--n_samples_z", type=int, default=n_samples_z)
parser.add_argument("--encoder_layer_sizes", type=list, default=[(lidar_input_size*n_samples_y), 256, 256])
parser.add_argument("--decoder_layer_sizes", type=list, default=[(joint_input_size*n_samples_y), 256, 256])
parser.add_argument("--latent_size", type=int, default=lidar_input_size*clusters)
parser.add_argument("--print_every", type=int, default=1000)
parser.add_argument("--fig_root", type=str, default='figs')
parser.add_argument("--conditional", action='store_true')
parser.add_argument("--num_labels", type=int, default=0)
parser.add_argument("--ckpt_dir", type=str, default="../experiments/ckpt_cvae_gmm/")
parser.add_argument("--ckpt", type=str, default="ckpt_cvae_gmm.pth")
parser.add_argument("--split", type=str, default="train")

args = parser.parse_args()

if __name__ == '__main__':
	if args.split == "train":
		train(args)
	else:
		test(args)
