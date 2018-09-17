import py_at_broker as pab
import numpy as np
import time


b = pab.broker()
print(b.register_signal("franka_target_pos", pab.MsgType.target_pos))
time.sleep(0.5)
print(b.request_signal("franka_state", pab.MsgType.franka_state))

counter = 0

def default_pos():
    pos_c = np.array([ 6.23963393e-01, -5.11126213e-06,  5.59791033e-01])
    msg = pab.target_pos_msg()
    msg.set_ctrl_t(pab.CtrlType.Cartesian)
    msg.set_pos(pos_c)
    msg.set_timestamp(time.clock_gettime(time.CLOCK_MONOTONIC))
    msg.set_fnumber(counter)
    msg.set_time_to_go(1.)
    b.send_msg("franka_target_pos", msg)
    time.sleep(0.5)
    return None

state = b.recv_msg("franka_state", -1)   
current_pos_c = state.get_c_pos()
print('current pos: {}'.format(current_pos_c))

n_points = 10
n_runs = 0
time2go = 1.
R = 0.1
delta_z = 2 * R / n_points
delta_theta = 2 * np.pi / n_points

while True:
    default_pos()
    counter += 1
    print(counter)

    for z in np.arange(current_pos_c[2], current_pos_c[2] + R, delta_z):
        for theta in np.arange(0, 2*np.pi, delta_theta):

            phi = np.arccos((z - current_pos_c[2]) / R)
            print(theta, z, phi)

            pos_c = np.array([R*np.cos(theta)*np.cos(phi), R*np.sin(theta)*np.cos(phi), R*np.sin(phi)])

            msg = pab.target_pos_msg()
            msg.set_ctrl_t(pab.CtrlType.Cartesian)
            msg.set_pos(pos_c)
            msg.set_timestamp(time.clock_gettime(time.CLOCK_MONOTONIC))
            msg.set_fnumber(counter)
            msg.set_time_to_go(time2go)

            b.send_msg("franka_target_pos", msg)
            time.sleep(0.5)
            new_state = b.recv_msg("franka_state", -1)
            print('next pos: {}'.format(new_state.get_c_pos()))
            counter += 1
            print(counter)
