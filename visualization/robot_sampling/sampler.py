import time
import sys
import lib
import numpy as np

def main():
    broker = lib.default()
    print()
    print("start collect_routine")
    counter = 0
    img_counter = 0
    default_pos_c = np.array([0.62, 0.00, 0.56])

    n_points = 100
    n_runs   = 1
    time2go  = 3.
    delta    = 2 * np.pi / n_points
    R        = 0.2
    save_filename = 'data_'

    threshold=0.95

    data = {}
    n_samples=1

    save_every = 100
    debug = False
    total_time = 20 # seconds

    i = 0
    runs = 0
    data = lib.init_data_run(data, runs)
    while i < 10e5:  
        try:
            start = time.time()
            # request data
            lidar     = broker.recv_msg("franka_lidar", -1)
            new_state = broker.recv_msg("franka_state", -1)
            #img       = broker.recv_msg("realsense_images", -1)
            # update data for lidar and state
            data = lib.update_data(data, lidar, new_state, runs)
            # save images
            #img_counter = lib.collect_rgb_depth(img, img_counter)

            print("\rtimesteps: {}".format(i), end="")
            sys.stdout.flush()
            i +=1

            if i % save_every == 0:
                lib.save_data(data, save_filename, runs)

        except KeyboardInterrupt:
            lib.save_data(data, save_filename, runs)
            print("input 0 to break")
            i = input()
            if i == "0": 
                break 
            else: continue

    lib.save_data(data, save_filename, runs)

if __name__ == '__main__':
    main()
