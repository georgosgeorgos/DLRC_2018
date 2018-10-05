import time

import lib
import numpy as np

counter = 0
img_counter = 0
default_pos_c = np.array([0.62, 0.00, 0.56])

n_points = 100
n_runs   = 1
time2go  = 3.
delta    = 2 * np.pi / n_points
R        = 0.2
save_filename = 'data_'

threshold=0.98

data = {}
n_samples=1

save_every = 10
debug = False
total_time = 20 # seconds

runs = -1

def main():
    counter = 0
    img_counter = 0
    default_pos_c = np.array([0.62, 0.00, 0.56])

    n_points = 100
    n_runs   = 100
    time2go  = 3.
    delta    = 2 * np.pi / n_points
    R        = 0.2
    save_filename = 'data_'

    threshold=0.98

    data = {}
    n_samples=1

    save_every = 10
    debug = False
    total_time = 20 # seconds

    runs = -1
    # initialize broker
    broker = lib.default()
    counter = lib.pos_msg(default_pos_c, broker, counter)
    print("Press ENTER")
    i = input()

    while runs < n_runs:    
        runs +=1
        print("RUN:", runs)
        data = lib.init_data_run(data, runs)
        try:
            time.sleep(2.0)
            start = time.time()

            while time.time() - start < total_time:
                X, Y, Z, alpha = lib.sample()

                if np.random.random() < threshold:
                    pos = lib.build_position(X, Y, Z)
                else:
                    pos = lib.build_position_orientation(X, Y, Z, alpha)
                    print("Change angle: ", alpha)

                counter = lib.pos_msg(pos, broker, counter)
                time.sleep(0.5)

                # request data
                lidar     = broker.recv_msg("franka_lidar", -1)
                new_state = broker.recv_msg("franka_state", -1)
                img       = broker.recv_msg("realsense_images", -1)

                # update data for lidar and state
                data = lib.update_data(data, lidar, new_state, runs)
                # save images
                img_counter = lib.collect_rgb_depth(img, img_counter)

                if debug == True:
                    print('next pos: {}'.format(new_state.get_c_pos()))

            if runs % save_every == 0:
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