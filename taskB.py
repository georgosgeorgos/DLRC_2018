import py_at_broker as pab
import numpy as np
import time


b = pab.broker()
print(b.register_signal("franka_target_pos", pab.MsgType.target_pos))
print(b.request_signal("franka_state", pab.MsgType.franka_state))

counter = 0
state = b.recv_msg("franka_state", -1)    
current_pos_c = state.get_c_pos()
print('current pos: {}'.format(current_pos_c))

n_points = 10
n_runs = 3
time2go = 1.
delta = 2 * np.pi / n_points
R = 0.2

for theta in np.arange(0, 2*n_runs*np.pi, delta):

    pos_c = current_pos_c + [R*np.sin(theta), R*np.cos(theta), 0]

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
