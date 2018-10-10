import time
import lib
import numpy as np
import sys

def main(broker):
    print()
    print("start random_trajectory")
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
    counter = lib.pos_msg(default_pos_c, broker, counter)
    print("Press ENTER")
    i = input()
    state = broker.recv_msg("franka_state", -1)
    joint_state = np.array(state.get_j_pos())
    pos_ = joint_state.copy()
    flag = True
    while runs < n_runs:
        runs += 1
        print()
        print("RUN:", runs)
        try:
            time.sleep(0.5)
            start = time.time()
            while flag:
                pos_[5] -= 0.5
                pos_[3] -= 0.5
                pos_[1] += 0.5
                counter = lib.pos_j_msg(pos_, broker, counter)
                time.sleep(2.0)
                flag = False

            while time.time() - start < total_time:
                X, Y, Z, alpha = lib.sample()

                if np.random.random() < threshold:
                    pos = lib.build_position(0.8, Y, Z)
                else:
                    pos = lib.build_position_orientation(X, Y, Z, alpha)
                    print("Change angle: ", alpha)
                    state = broker.recv_msg("franka_state", -1)
                    joint_state = np.array(state.get_j_pos())
                    joint_state[5] = pos_[5]
                    joint_state[3] = pos_[3]
                    joint_state[1] = pos_[1]
                    counter = lib.pos_j_msg(joint_state, broker, counter)
                    time.sleep(0.5)

                counter = lib.pos_msg(pos, broker, counter)
                #counter = lib.pos_j_msg(joint_state, broker, counter)
                time.sleep(0.5)
                print("\rposition counter: {}".format(counter), end="")
                sys.stdout.flush()

        except KeyboardInterrupt:
            break
if __name__ == '__main__':
    broker = lib.default()
    main(broker)
