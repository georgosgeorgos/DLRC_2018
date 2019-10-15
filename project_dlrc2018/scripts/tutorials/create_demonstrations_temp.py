import os.path as osp
import sys
import time
import pickle as pkl
import matplotlib.pyplot as plt
from collections import defaultdict

from scripts.collect_data.lib import *

data = defaultdict()
rootdir = "../"
data_path = osp.join(rootdir, "data")
robot_name = "franka"
max_sync_jitter = 0.2

b = default()

counter = 0
time.sleep(1)  # Allow some time for the communication setup

num_demonstrations = input("Enter number of demonstrations to record: ")
num_demonstrations = int(num_demonstrations)
num_timesteps = input("Enter number of time steps for one demonstration: ")
num_timesteps = int(num_timesteps)

for d in range(num_demonstrations):
    data = init_data_run(data, d)
    print("Demo: {:d}".format(d))

    start = time.time()
    for t in range(num_timesteps):
        # Receive franka state
        recv_start = time.clock_gettime(time.CLOCK_MONOTONIC)
        msg_panda = b.recv_msg(robot_name + "_state", -1)
        recv_stop = time.clock_gettime(time.CLOCK_MONOTONIC)

        # Create and fill message
        counter = tau_msg(b, counter)

        print("\rtimesteps: {}".format(t), end="")
        sys.stdout.flush()

    end = time.time()
