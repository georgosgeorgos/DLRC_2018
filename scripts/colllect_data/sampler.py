import time
import sys
import lib
import numpy as np

def main(broker):
    print()
    print("start collect_routine")
    data          = {}
    img_counter   = 0
    save_every    = 1000
    save_filename = 'data_'
    #________________________________________________________________
    data = lib.init_data_run(data, 0)
    for i in range(10e5):  
        try:
    #________________________________________________________________
            start = time.time()
            # request data
            lidar     = broker.recv_msg("franka_lidar", -1)
            new_state = broker.recv_msg("franka_state", -1)
            #img       = broker.recv_msg("realsense_images", -1)
            # update data for lidar and state
            data = lib.update_data(data, lidar, new_state, 0)
            # save images
            #img_counter = lib.collect_rgb_depth(img, img_counter)
            print("\rtimesteps: {}".format(i), end="")
            sys.stdout.flush()
            if i % save_every==0: 
                lib.save_data(data, save_filename, 0)
    #_______________________________________________________________
        except KeyboardInterrupt:
            lib.save_data(data, save_filename, 0)
            print("press 0 to break")
            b=input()
            if b=="0": 
                break 
            else: continue
    #________________________________________________________________
    lib.save_data(data, save_filename, 0)

if __name__ == '__main__':
    main()