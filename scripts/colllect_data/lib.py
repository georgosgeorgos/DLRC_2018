import pickle as pkl
import time
from os import makedirs
from os.path import exists
import numpy as np
import py_at_broker as pab
from PIL import Image
from libtiff import TIFF
from pyquaternion import Quaternion

def path_exists(path):
    if not exists(path):
        makedirs(path)
    return path

def default():
    broker = pab.broker()
    print(broker.register_signal("franka_target_pos", pab.MsgType.target_pos))
    time.sleep(0.5)
    print(broker.request_signal("franka_lidar", pab.MsgType.franka_lidar))
    time.sleep(0.5)
    print(broker.request_signal("franka_state", pab.MsgType.franka_state))
    time.sleep(0.5)
    #print("Register signal <_des_tau> {}".format(broker.register_signal("franka_des_tau", pab.MsgType.des_tau)))
    #time.sleep(0.5)
    # print(broker.request_signal("realsense_images", pab.MsgType.realsense_image))
    # time.sleep(0.5)
    return broker

def pos_msg(pos, broker, counter, time2go=4):
    msg = pab.target_pos_msg()
    msg.set_ctrl_t(pab.CtrlType.Cartesian)
    msg.set_pos(pos)
    msg.set_timestamp(time.clock_gettime(time.CLOCK_MONOTONIC))
    msg.set_fnumber(counter)
    msg.set_time_to_go(time2go)
    broker.send_msg("franka_target_pos", msg)
    counter += 1
    return counter

def pos_j_msg(pos, broker, counter, time2go=4):
    msg = pab.target_pos_msg()
    msg.set_ctrl_t(pab.CtrlType.Joint)
    msg.set_pos(pos)
    msg.set_timestamp(time.clock_gettime(time.CLOCK_MONOTONIC))
    msg.set_fnumber(counter)
    msg.set_time_to_go(time2go)
    broker.send_msg("franka_target_pos", msg)
    counter += 1
    return counter

def tau_msg(broker, counter):
    msg = pab.des_tau_msg()
    msg.set_timestamp(time.clock_gettime(time.CLOCK_MONOTONIC))
    msg.set_fnumber(counter)
    msg.set_j_torque_des(np.zeros(7))
    # Send message
    broker.send_msg("franka_des_tau", msg)
    counter += 1
    return counter

def build_init_j(broker):
    state = broker.recv_msg("franka_state", -1)
    joint_state = np.array(state.get_j_pos())
    pos_ = joint_state.copy()
    pos_[5] -= 0.5
    pos_[3] -= 0.5
    pos_[1] += 0.5
    return pos_

def build_generic_j(broker, pos_):
    state = broker.recv_msg("franka_state", -1)
    joint_state = np.array(state.get_j_pos())
    joint_state[5] = pos_[5] * np.random.random()
    joint_state[3] = pos_[3] * np.random.random()
    joint_state[1] = pos_[1] * np.random.random()
    return joint_state

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

def sample():
    #WallMin = [0.28, -0.78, 0.02]
    #WallMax = [0.82,  0.78, 1.08]
    WallMin = [0.38, -0.58, 0.06]
    WallMax = [0.76,  0.58, 0.8]

    X = np.random.uniform(WallMin[0], WallMax[0], 1)[0]
    Y = np.random.uniform(WallMin[1], WallMax[1], 1)[0]
    Z = np.random.uniform(WallMin[2], WallMax[2], 1)[0]
    alpha = np.random.uniform(-1, 1, 1)[0]
    return X, Y, Z, alpha


def collect_rgb_depth(img, img_counter, path_rgb='./RGB/', path_depth='./DEPTH/'):

    rgb = np.reshape(img.get_rgb(), img.get_shape_rgb())
    rgb = Image.fromarray(rgb)
    path_exists(path_rgb)
    rgb.save(path_rgb + str(img_counter) + '.png')

    path_exists(path_depth)
    depth = np.reshape(img.get_depth(), img.get_shape_depth())
    tiff = TIFF.open(path_depth + str(img_counter) + '.tiff', mode='w')
    tiff.write_image(depth)
    tiff.close()
    
    img_counter += 1
    return img_counter


def update_data(data, msg_lidar, msg_state, runs):
    data[runs]["trajectory"].append(list(msg_state.get_c_pos()))

    data[runs]["state"]["j_pos"].append(list(msg_state.get_j_pos()))
    data[runs]["state"]["j_vel"].append(list(msg_state.get_j_vel()))
    data[runs]["state"]["j_load"].append(list(msg_state.get_j_load()))
    data[runs]["state"]["c_pos"].append(list(msg_state.get_c_pos()))
    data[runs]["state"]["c_vel"].append(list(msg_state.get_c_vel()))
    data[runs]["state"]["c_ori_quat"].append(list(msg_state.get_c_ori_quat()))
    data[runs]["state"]["dc_ori_quat"].append(list(msg_state.get_dc_ori_quat()))
    data[runs]["state"]["timestamp"].append(msg_state.get_timestamp())
    
    data[runs]["lidar"]["measurements"].append(list(msg_lidar.get_data()))
    data[runs]["lidar"]["timestamp"].append(msg_lidar.get_timestamp())

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
                            "measurements": [], "timestamp": []
                            }
                  }
    return data

def save_data(data, save_filename, runs):
    with open(save_filename + str(runs) + '.pkl', 'wb') as f:
            pkl.dump(data, f, protocol=pkl.HIGHEST_PROTOCOL)
    print("data saved")
