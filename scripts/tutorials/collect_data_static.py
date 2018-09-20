import py_at_broker as pab
from PIL import Image
from libtiff import TIFF
import numpy as np
import matplotlib.pyplot as plt
import time

b = pab.broker()
print(b.request_signal("franka_lidar", pab.MsgType.franka_lidar))
print(b.request_signal("realsense_images", pab.MsgType.realsense_image))

counter = 0

start = time.time()
lidar_list = []
img_counter = 0

time2go = 1.

lidar_list = []
c = 0
n_samples=50
img_counter = 0

for n in range(10):

    print("press Enter")
    i = input()
    try:
        c = 0
        while c < n_samples:
            lidar = b.recv_msg("franka_lidar", -1)
            lidar_list.append(lidar.get_data())
            counter += 1
            c += 1

        print("end lidar data collected")
        img = b.recv_msg("realsense_images", -1)

        rgb = np.reshape(img.get_rgb(), img.get_shape_rgb())
        rgb = Image.fromarray(rgb)
        #rgb.save('./images_self/rgb/' + str(img_counter) + '.png')

        depth = np.reshape(img.get_depth(), img.get_shape_depth())
        #tiff = TIFF.open('./images_self/depth/' + str(img_counter) + '.tiff', mode='w')
        #tiff.write_image(depth)
        #tiff.close()

        lidar_list.append(lidar.get_data())
        counter += 1
        img_counter += 1
    except KeyboardInterrupt:
        break


lidar_array = np.array(lidar_list)
np.save("lidar_data_self.npy", lidar_array)