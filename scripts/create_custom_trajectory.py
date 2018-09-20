import py_at_broker as pab
import numpy as np
import matplotlib.pyplot as plt
import time
from collections import defaultdict
import sys
import pickle as pkl
import os.path as osp

# dictionary with key=trajectory and value list of measurements over `num_timesteps`
data = defaultdict()
lidar_id = 3 # zero-indexed, 9 in total
data_path = '../../data'

# initialization broker
b = pab.broker()
print(b.request_signal("franka_lidar", pab.MsgType.franka_lidar))

num_trajectories = input("Enter number of trajectories: ")
num_trajectories = int(num_trajectories)

num_timesteps = input("Enter number of timesteps per trajectory: ")
num_timesteps = int(num_timesteps)

for it in range(num_trajectories):
    data[it] = defaultdict()
    data[it]['data'] = []
    data[it]['lidar'] = lidar_id
    # create trajectories by manually moving the robot arm
    print("---- trajectory: {:d}".format(it))

    start = time.time()

    # collect measurements
    for t in range(num_timesteps):
        lidar = b.recv_msg("franka_lidar", -1)
        measurement = lidar.get_data()[lidar_id]
        data[it]['data'].append(measurement)
        print("\rmeasurements: {}\ttimesteps: {}".format(measurement, t), end="")
        sys.stdout.flush()

    end = time.time()
    print("\n### Trajectory took {:.4f} seconds.".format(end-start))

    data[it]['frequency'] = num_timesteps / (end - start)

    # plot some things!
    plt.plot(data[it]['data'])
    plt.show()
    plt.hist(data[it]['data'])
    plt.show()


# save collected data
print("Saving...")
pkl.dump(data, open(osp.join(data_path, 'custom_trajectories.pkl'), "wb"))
        