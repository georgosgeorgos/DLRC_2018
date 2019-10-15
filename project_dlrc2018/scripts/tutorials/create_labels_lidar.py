import time

import matplotlib.pyplot as plt
import numpy as np
import py_at_broker as pab

b = pab.broker()
print(b.request_signal("franka_lidar", pab.MsgType.franka_lidar))

counter = 0

start = time.time()
lidar_list = []
runs = 0
while True:
    try:
        c = 0
        print("Press Enter to start")
        print("Run:", runs)
        runs += 1
        i = input()
        while c < 100:
            lidar = b.recv_msg("franka_lidar", -1)
            lidar_list.append(lidar.get_data()[3])
            counter += 1
            print(counter)
            c += 1

        print(lidar_list)
        lidar_array = np.array(lidar_list)
        lidar_array[lidar_array > 1900] = 2000
        print(lidar_array[-100:].mean())
        print(lidar_array[-100:].std())
        plt.plot(lidar_array[-100:])
        plt.show()
        plt.hist(lidar_array[-100:])
        plt.show()
    except KeyboardInterrupt:
        print("save")
        lidar_array = np.array(lidar_list)
        ##np.save("lidar_array_static.npy", lidar_array)
        break
