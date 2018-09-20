import py_at_broker as pab
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from libtiff import TIFF
import pickle as pkl
from pyquaternion import Quaternion

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

def pos_j_msg(pos):
    msg = pab.target_pos_msg()
    msg.set_ctrl_t(pab.CtrlType.Joint)
    msg.set_pos(pos)
    msg.set_timestamp(time.clock_gettime(time.CLOCK_MONOTONIC))
    msg.set_fnumber(counter)
    msg.set_time_to_go(time2go)
    b.send_msg("franka_target_pos", msg)

state = b.recv_msg("franka_state", -1)    
current_pos_c = state.get_c_pos()
print('initial pos: {}'.format(current_pos_c))

n_points = 100
n_runs   = 1
time2go  = 3.
delta    = 2 * np.pi / n_points
R        = 0.2
save_filename = 'data_5_min_static'

data = {}
c = 0
n_samples=1


WallMin = [0.28, -0.78, 0.02]
WallMax = [0.82,  0.78, 1.08]

X = np.random.uniform(WallMin[0], WallMax[0], 1)[0]
Y = np.random.uniform(WallMin[1], WallMax[1], 1)[0]
Z = np.random.uniform(WallMin[2], WallMax[2], 1)[0]

alpha = np.random.uniform(-0.01, 0.01, 1)[0]

runs = -1
pos_msg(default_pos_c)
counter += 1
print("Press ENTER")
i = input()

save_every = 10
debug = False
total_time = 300 # seconds

while runs < n_runs:    
    runs +=1
    print("RUN:", runs)
    data[runs] = {"trajectory": [], 
                  "state": {"j_pos": [], "j_vel": [], "j_load": [], 
                  "c_pos": [], "c_vel": [], "c_ori_quat": [], 
                  "dc_ori_quat": [], "timestamp": []}, 
                  "lidar": {"measure": [], "timestamp": []}}
    try:
        state = b.recv_msg("franka_state", -1)    
        pos_c = state.get_c_pos()
        #print('current pos: {}'.format(pos_c))
        counter += 1
        time.sleep(2.0)
        start = time.time()
        while time.time() - start < total_time:
            X = np.random.uniform(WallMin[0], WallMax[0], 1)[0]
            Y = np.random.uniform(WallMin[1], WallMax[1], 1)[0]
            Z = np.random.uniform(WallMin[2], WallMax[2], 1)[0]

            alpha = np.random.uniform(-0.01, 0.01, 1)[0]
            #q = list(Quaternion.random())
            #if np.random.random() < 1.1:
            pos = np.array([X, Y, Z])
            pos_msg(pos)
            # else:
            #     print("Change angle: ", alpha)
            #     axis = np.zeros(3)
            #     a = np.argmax(np.random.random(3))
            #     axis[a] = 1
            #     q = Quaternion(axis=axis, angle=alpha)
            #     q = list(q)
            #     pos = np.array([X, Y, Z] + q)
            #     pos_msg(pos)

            counter += 1
            time.sleep(0.5)
            
            lidar = b.recv_msg("franka_lidar", -1)
            counter += 1

            new_state = b.recv_msg("franka_state", -1)
            counter += 1

            data[runs]["trajectory"].append(list(new_state.get_c_pos()))

            data[runs]["state"]["j_pos"].append(list(new_state.get_j_pos()))
            data[runs]["state"]["j_vel"].append(list(new_state.get_j_vel()))
            data[runs]["state"]["j_load"].append(list(new_state.get_j_load()))
            data[runs]["state"]["c_pos"].append(list(new_state.get_c_pos()))
            data[runs]["state"]["c_vel"].append(list(new_state.get_c_vel()))
            data[runs]["state"]["c_ori_quat"].append(list(new_state.get_c_ori_quat()))
            data[runs]["state"]["dc_ori_quat"].append(list(new_state.get_dc_ori_quat()))
            data[runs]["state"]["timestamp"].append(new_state.get_timestamp())
            
            data[runs]["lidar"]["measure"].append(list(lidar.get_data()))
            data[runs]["lidar"]["timestamp"].append(lidar.get_timestamp())

            if debug == True:
                print('next pos: {}'.format(new_state.get_c_pos()))

        if runs % save_every == 0:
            with open(save_filename + str(runs) +'.pkl', 'wb') as f:
                pkl.dump(data, f, protocol=pkl.HIGHEST_PROTOCOL)

    except KeyboardInterrupt:
        with open(save_filename + str(runs) +'.pkl', 'wb') as f:
            pkl.dump(data, f, protocol=pkl.HIGHEST_PROTOCOL)

        print("input 0 to break")
        i = input()
        if i == "0":
            break
        else:
            continue

    with open(save_filename + str(runs) + '.pkl', 'wb') as f:
        pkl.dump(data, f, protocol=pkl.HIGHEST_PROTOCOL)