import time
import sys
import lib
import numpy as np
import matplotlib.pyplot as plt

def main(broker):
    print()
    print("start collect_routine")
    data          = {}
    img_counter   = 0
    save_every    = 1000
    save_filename = 'data_'
    i=0
    k = 1
    #________________________________________________________________
    data = lib.init_data_run(data, k)
    start = time.time()
    lidars = []
    while i < 10e7:
        print("ok")
        try:
            print("ok")
    #________________________________________________________________
            # request data
            #lidar     = broker.recv_msg("franka_lidar", -1)
            #print(lidar)
            new_state = broker.recv_msg("franka_state", -1)
            img       = broker.recv_msg("realsense_images", -1)
            # update data for lidar and state
            #data = lib.update_data(data, lidar, new_state, k)
            # save images
            img_counter = lib.collect_rgb_depth(img, img_counter)

            print("\rtimesteps: {}".format(i), end="")
            sys.stdout.flush()
            if i % save_every==0 and i !=0: 
                lib.save_data(data, save_filename, k)   
            i +=1
    #_______________________________________________________________
        except KeyboardInterrupt:
            lib.save_data(data, save_filename, k)
            print("press 0 to break")
            b=input()
            if b=="0": 
                break 
            else: continue
    #________________________________________________________________
    lib.save_data(data, save_filename, k)

if __name__ == '__main__':
    main()
