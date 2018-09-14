import py_at_broker as pab
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from libtiff import TIFF

b = pab.broker()
print(b.register_signal("franka_target_pos", pab.MsgType.target_pos))
time.sleep(0.5)
print(b.request_signal("franka_lidar", pab.MsgType.franka_lidar))
print(b.request_signal("franka_state", pab.MsgType.franka_state))
time.sleep(0.5)

counter = 0
default_pos_c = np.array([0.62, 0.00, 0.56])

def pos_msg(pos):
    msg = pab.target_pos_msg()
    msg.set_ctrl_t(pab.CtrlType.Cartesian)
    msg.set_pos(pos)
    msg.set_timestamp(time.clock_gettime(time.CLOCK_MONOTONIC))
    msg.set_fnumber(counter)
    msg.set_time_to_go(time2go)
    b.send_msg("franka_target_pos", msg)

state = b.recv_msg("franka_state", -1)    
current_pos_c = state.get_c_pos()
print('initial pos: {}'.format(current_pos_c))

n_points = 100
n_runs = 10
time2go = 1.
delta = 2 * np.pi / n_points
R = 0.2

lidar_dict = {}
configuration_dict = {}
c = 0
n_samples=1


WallMin = [0.28, -0.78, 0.02]
WallMax = [0.82,  0.78, 1.08]

X = np.random.uniform(WallMin[0], WallMax[0], 100)
Y = np.random.uniform(WallMin[1], WallMax[1], 100)
Z = np.random.uniform(WallMin[2], WallMax[2], 100)

runs = -1
import pickle as pkl
while runs < 10:
    print("Press ENTER")
    print("RUN:", runs)
    i = input()
    runs +=1
    configuration_dict[runs] = []
    lidar_dict[runs] = []
    try:
        state = b.recv_msg("franka_state", -1)    
        pos_c = state.get_c_pos()
        print('current pos: {}'.format(pos_c))
        counter += 1
        time.sleep(2.0)
        for _ in range(300):
            # measure
            lidar = b.recv_msg("franka_lidar", -1)
            lidar_dict[runs].append(lidar.get_data())
            counter += 1

            new_state = b.recv_msg("franka_state", -1)
            counter += 1
            configuration_dict[runs].append([new_state.get_c_pos(), new_state.get_c_vel(), 
                                 new_state.get_c_ori_quat(), new_state.get_dc_ori_quat(), 
                                 new_state.get_j_pos(), new_state.get_j_vel(), new_state.get_j_load()])

            print('next pos: {}'.format(new_state.get_c_pos()))
            counter += 1

    except KeyboardInterrupt:
        #configuration_array = np.array(configuration_dict)
        #lidar_array = np.array(lidar_list)
        #np.save("./lidar_bad_conf.npy", lidar_array)

        with open('lidars.pickle', 'wb') as f:
            pkl.dump(lidar_dict, f, protocol=pkl.HIGHEST_PROTOCOL)

        with open('configurations.pickle', 'wb') as f:
            pkl.dump(configuration_dict, f, protocol=pkl.HIGHEST_PROTOCOL)

        #np.save("./configuration.npy", configuration_dict)
        runs += 1

    #configuration_array = np.array(configuration_dict)
    #lidar_array = np.array(lidar_list)
    #np.save("./lidar_bad_conf.npy", lidar_array)

    with open('configurations.pickle', 'wb') as f:
        pkl.dump(configuration_dict, f, protocol=pkl.HIGHEST_PROTOCOL)

    with open('lidars.pickle', 'wb') as f:
            pkl.dump(lidar_dict, f, protocol=pkl.HIGHEST_PROTOCOL)

    #np.save("./configuration.npy", configuration_dict)



