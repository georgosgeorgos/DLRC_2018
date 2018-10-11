from time import sleep, time
import lib
import numpy as np
import sys

def main(broker):
    print()
    print("start random_trajectory")
    counter    = 0
    n_runs     = 360
    threshold  = 0.98
    total_time = 30 # seconds
    is_joint   = True
    #______________________________________________________
    default_pos_c = np.array([0.62, 0.00, 0.56])
    counter = lib.pos_msg(default_pos_c, broker, counter)
    print("Press ENTER")
    b=input()
    pos_    = lib.build_init_j(broker)
    counter=lib.pos_j_msg(pos_, broker, counter)
    #______________________________________________________
    for run in range(n_runs):
        print("RUN:", run)
        try:
            sleep(0.5)
            start=time()
            while time() - start < total_time:
                X, Y, Z, alpha=lib.sample()

                if np.random.random() < threshold:
                    pos=lib.build_position(X, Y, Z)
                else:
                    pos=lib.build_position_orientation(X, Y, Z, alpha)
                    if is_joint:
                        pos_j=lib.build_generic_j(broker, pos_)
                        counter=lib.pos_j_msg(pos_j, broker, counter)
                        sleep(0.5)

                counter=lib.pos_msg(pos, broker, counter)
                #counter = lib.pos_j_msg(joint_state, broker, counter)
                sleep(0.5)
                print("\rposition counter: {}".format(counter), end="")
                sys.stdout.flush()
        except KeyboardInterrupt:
            break
    #____________________________________________________
    
if __name__ =='__main__':
    broker=lib.default()
    main(broker)