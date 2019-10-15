import time

import numpy as np
import py_at_broker as pab
from PIL import Image
from libtiff import TIFF

b = pab.broker()
print(b.request_signal("franka_lidar", pab.MsgType.franka_lidar))
print(b.request_signal("realsense_images", pab.MsgType.realsense_image))

counter = 0

start = time.time()
lidar_list = []
img_counter = 0

time2go = 1.0

lidar_list = []
c = 0
n_samples = 10
img_counter = 0
runs = 0
for n in range(10):
    print("Run:", runs)
    runs += 1
    print("press Enter")
    i = input()
    try:
        c = 0
        while c < n_samples:
            counter += 1
            c += 1
            img = b.recv_msg("realsense_images", -1)

            rgb = np.reshape(img.get_rgb(), img.get_shape_rgb())
            rgb = Image.fromarray(rgb)
            rgb.save("./images_franka/rgb/" + str(img_counter) + ".png")

            depth = np.reshape(img.get_depth(), img.get_shape_depth())
            tiff = TIFF.open("./images_franka/depth/" + str(img_counter) + ".tiff", mode="w")
            tiff.write_image(depth)
            tiff.close()
            img_counter += 1
            print(counter)
    except KeyboardInterrupt:
        break
