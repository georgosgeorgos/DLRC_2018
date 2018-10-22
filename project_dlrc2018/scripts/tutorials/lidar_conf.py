import time

import matplotlib.pyplot as plt
import numpy as np
import py_at_broker as pab

b = pab.broker()
print(b.request_signal("franka_lidar", pab.MsgType.franka_lidar))

counter = 0

start = time.time()
lidar_list = []

while True:

    try:
        lidar = b.recv_msg("franka_lidar", -1)

        lidar_list.append(lidar.get_data())
        counter += 1
    except KeyboardInterrupt:
        break

    print(counter)


lidar_array = np.array(lidar_list)

for j in range(9):
    plt.plot(lidar_array[:,j])
    plt.title(j+1)
    plt.show()
    plt.hist(lidar_array[:,j])
    plt.title(j+1)
    plt.show()
