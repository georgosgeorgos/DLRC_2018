import py_at_broker as pab
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from libtiff import TIFF
#_________________________________________________________________________________________
b = pab.broker()
print(b.register_signal("franka_target_pos", pab.MsgType.target_pos))
time.sleep(0.5)
print(b.register_signal("franka_gripper", pab.MsgType.gripper_cmd))
time.sleep(0.5)
#_________________________________________________________________________________________
print(b.request_signal("franka_state", pab.MsgType.franka_state))
time.sleep(0.5)
#_________________________________________________________________________________________
counter = 0
default_pos_c = np.array([0.62, 0.00, 0.56])

def f_msg(pos):
    msg = pab.target_pos_msg()
    msg.set_ctrl_t(pab.CtrlType.Cartesian)
    msg.set_pos(pos)
    msg.set_timestamp(time.clock_gettime(time.CLOCK_MONOTONIC))
    msg.set_fnumber(counter)
    msg.set_time_to_go(time2go)
    b.send_msg("franka_target_pos", msg)

def g_msg(r):
    msg = pab.gripper_cmd_msg()
    msg.set_cmd_t(r)
    msg.set_width(1.)
    msg.set_speed(0.5)
    msg.set_force(2.)
    msg.set_epsilon_in(0.0001)
    msg.set_epsilon_out(0.0001)
    msg.set_timestamp(time.clock_gettime(time.CLOCK_MONOTONIC))
    msg.set_fnumber(counter)
    #msg.set_time_to_go(time2go)
    b.send_msg("franka_gripper", msg)

state = b.recv_msg("franka_state", -1)    
current_pos_c = state.get_c_pos()
print('initial pos: {}'.format(current_pos_c))

n_points = 10
n_runs   = 20
time2go  = 1.
delta    = 2 * np.pi / n_points
R        = 0.2

lidar_list  = []
n_samples   = 50
img_counter = 0
delta_z = R / n_points
delta_theta = 2 * np.pi / n_points
r = 2
while True:
    try:
        f_msg(default_pos_c)
        time.sleep(0.5)
        state = b.recv_msg("franka_state", -1)    
        current_pos_c = state.get_c_pos()

        print("default pos: {}".format(default_pos_c))
        print('current pos: {}'.format(current_pos_c))
        counter += 1
        time.sleep(2.0)
        for n in range(n_runs):
            i = 0
            print("RUN:", n)
            print()
            for z in np.arange(current_pos_c[2], current_pos_c[2] + R, delta_z):
                print("ZZZ", z, i)
                i +=1
                phi = np.arccos((z - current_pos_c[2]) / R)
                print("grasp")
                g_msg(r)
                if r == 2:
                    r = 1
                else:
                    r = 2
                time.sleep(0.5)

                for theta in np.arange(0, 2*np.pi, delta_theta):
                    #print(theta, z, phi)
                    pos_c = current_pos_c + np.array([R*np.cos(theta)*np.cos(phi), R*np.sin(theta)*np.cos(phi), R*np.sin(phi)])
                    
                    # move
                    print("move")
                    f_msg(pos_c)
                    time.sleep(0.5)
                    new_state = b.recv_msg("franka_state", -1)
                    print('next pos: {}'.format(new_state.get_c_pos()))
                    counter += 1

            for z in np.arange(current_pos_c[2] + R, current_pos_c[2], -delta_z):
                print("ZZZ", z, i)
                phi = np.arccos((z - current_pos_c[2]) / R)
                print("grasp")
                g_msg(r)
                if r == 2:
                    r = 1
                else:
                    r = 2
                time.sleep(0.5)
                for theta in np.arange(0, 2*np.pi, delta_theta):
                    print(theta, z, phi)
                    pos_c = current_pos_c - np.array([R*np.cos(theta)*np.cos(phi), R*np.sin(theta)*np.cos(phi), R*np.sin(phi)])
                    
                    # move
                    print("move")
                    f_msg(pos_c)
                    time.sleep(0.5)

                    new_state = b.recv_msg("franka_state", -1)
                    print('next pos: {}'.format(new_state.get_c_pos()))
                    counter += 1
    except KeyboardInterrupt:
        break