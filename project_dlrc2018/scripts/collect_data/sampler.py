import time
import sys
import lib
import numpy as np
import matplotlib.pyplot as plt


def main(broker):
    print()
    print("start collect_routine")
    data = {}
    img_counter = 0
    save_every = 1000
    save_filename = "data_"
    i = 0
    # ________________________________________________________________
    data = lib.init_data_run(data, 0)
    start = time.time()
    lidars = []
    while True:
        try:
            # ________________________________________________________________
            # request data
            lidar = broker.recv_msg("franka_lidar", -1)
            new_state = broker.recv_msg("franka_state", -1)
            img = broker.recv_msg("realsense_images", -1)
            lidars.append(list(lidar.get_data()))
            # update data for lidar and state
            data = lib.update_data(data, lidar, new_state, 0)
            # save images
            img_counter = lib.collect_rgb_depth(img, img_counter)
            print("\rtimesteps: {}".format(i), end="")
            sys.stdout.flush()
            if i % save_every == 0 and i != 0:
                lib.save_data(data, save_filename, 0)
            i += 1
        # _______________________________________________________________
        except KeyboardInterrupt:
            lib.save_data(data, save_filename, 0)
            print("press 0 to break")
            b = input()
            if b == "0":
                break
            else:
                continue
    # ________________________________________________________________
    lib.save_data(data, save_filename, 0)


if __name__ == "__main__":
    main()
