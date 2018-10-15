import os.path as osp
import sys
import matplotlib.pyplot as plt
from collections import defaultdict

from scripts.scripts_collect_train_data.lib import *

data = defaultdict()
rootdir = '../'
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

        # Check that the messages we're getting are not too old, reset the
        # entire script if we get out of sync
        if (2 * recv_stop - recv_start - msg_panda.get_timestamp()) > max_sync_jitter:
            print("De-synced, messages too old\n Resetting...")

        # Create and fill message
        counter = tau_msg(b, counter)
        msg_lidar = b.recv_msg(robot_name + "_lidar", -1)

        data = update_data(data, msg_lidar, msg_panda, d)

        print("\rtimesteps: {}".format(t), end="")
        sys.stdout.flush()

    end = time.time()
    print("\n### Demo took {:.4f} seconds.".format(end-start))
    data[d]["lidar"]['freq'] = num_timesteps / (end - start)

    plt.plot(data[d]["lidar"]["measurements"])
    plt.show()
    plt.hist(data[d]["lidar"]["measurements"])
    plt.show()

# save collected data
print("Saving...")
pkl.dump(data, open(osp.join(data_path, "cluster.pkl"), "wb"))