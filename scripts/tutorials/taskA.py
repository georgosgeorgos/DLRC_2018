import time

import matplotlib.pyplot as plt
import numpy as np
import py_at_broker as pab

b = pab.broker()
print(b.request_signal("franka_lidar", pab.MsgType.franka_lidar))
print(b.request_signal("realsense_images", pab.MsgType.realsense_image))

counter = 0

start = time.time()
lidar_list = []

while True:
    if counter > 5:
        break

    img = b.recv_msg("realsense_images", -1)
    lidar = b.recv_msg("franka_lidar", -1)

    lidar_list.append(lidar.get_data())

    image = np.reshape(img.get_rgb(), img.get_shape_rgb())
    plt.imshow(image)
    plt.show()
    counter += 1

## 1 sample/0.06 s
end = time.time()
print(end-start)
lidar_array = np.array(lidar_list)
print(lidar_array.shape)

for j in range(9):
    plt.plot(lidar_array[:,j])

plt.show()
