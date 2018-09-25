import py_at_broker as pab
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from libtiff import TIFF
import pickle as pkl
from pyquaternion import Quaternion

def default():
    broker = pab.broker()
    print(broker.register_signal("franka_target_pos", pab.MsgType.target_pos))
    time.sleep(0.5)
    print(broker.request_signal("franka_lidar", pab.MsgType.franka_lidar))
    time.sleep(0.5)
    print(broker.request_signal("franka_state", pab.MsgType.franka_state))
    time.sleep(0.5)
    print(broker.request_signal("realsense_images", pab.MsgType.realsense_image))
    time.sleep(0.5)
    return broker

def pos_msg(pos, broker, counter, time2go=3):
    msg = pab.target_pos_msg()
    msg.set_ctrl_t(pab.CtrlType.Cartesian)
    msg.set_pos(pos)
    msg.set_timestamp(time.clock_gettime(time.CLOCK_MONOTONIC))
    msg.set_fnumber(counter)
    msg.set_time_to_go(time2go)
    broker.send_msg("franka_target_pos", msg)
    counter += 1
    return counter

def build_position(X,Y,Z):
    pos = np.array([X, Y, Z])
    return pos

def build_position_orientation(X, Y, Z, alpha):
    axis = np.zeros(3)
    a = np.argmax(np.random.random(3))
    axis[a] = 1
    q = Quaternion(axis=axis, angle=alpha)
    q = list(q)
    pos = np.array([X, Y, Z] + q)
    return pos

# def pos_j_msg(pos, broker):
#     msg = pab.target_pos_msg()
#     msg.set_ctrl_t(pab.CtrlType.Joint)
#     msg.set_pos(pos)
#     msg.set_timestamp(time.clock_gettime(time.CLOCK_MONOTONIC))
#     msg.set_fnumber(counter)
#     msg.set_time_to_go(time2go)
#     broker.send_msg("franka_target_pos", msg)

def sample():
    #WallMin = [0.28, -0.78, 0.02]
    #WallMax = [0.82,  0.78, 1.08]
    WallMin = [0.38, -0.58, 0.2]
    WallMax = [0.68,  0.58, 0.8]

    X = np.random.uniform(WallMin[0], WallMax[0], 1)[0]
    Y = np.random.uniform(WallMin[1], WallMax[1], 1)[0]
    Z = np.random.uniform(WallMin[2], WallMax[2], 1)[0]
    alpha = np.random.uniform(-0.01, 0.01, 1)[0]
    return X, Y, Z, alpha

def collect_rgb_depth(img, img_counter):
    rgb = np.reshape(img.get_rgb(), img.get_shape_rgb())
    rgb = Image.fromarray(rgb)
    rgb.save('./RGB/' + str(img_counter) + '.png')

    depth = np.reshape(img.get_depth(), img.get_shape_depth())
    tiff = TIFF.open('./DEPTH/' + str(img_counter) + '.tiff', mode='w')
    tiff.write_image(depth)
    tiff.close()
    img_counter += 1
    return img_counter

def update_data(data, lidar, new_state, runs):
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
    return data

def init_data_run(data, runs):
    data[runs] = {
                  "trajectory": [], 
                  "state": {
                            "j_pos": [], "j_vel": [], "j_load": [], 
                            "c_pos": [], "c_vel": [], "c_ori_quat": [], 
                            "dc_ori_quat": [], "timestamp": []
                            }, 
                  "lidar": {
                            "measure": [], "timestamp": []
                            }
                  }
    return data

def save_data(data, save_filename, runs):
    with open(save_filename + str(runs) + '.pkl', 'wb') as f:
            pkl.dump(data, f, protocol=pkl.HIGHEST_PROTOCOL)
    print("data saved")
