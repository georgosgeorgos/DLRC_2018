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


delta = 2 * np.pi / 10

for theta in np.arange(0, 6*np.pi, delta):

    pos_c = current_pos_c + [0.2*np.sin(theta), 0.2*np.cos(theta), 0]

    msg = pab.target_pos_msg()
    msg.set_ctrl_t(0)
    msg.set_pos(pos_c)
    msg.set_timestamp(time.clock_gettime(time.CLOCK_MONOTONIC))
    msg.set_fnumber(counter)
    msg.set_time_to_go(1.)

    b.send_msg("franka_target_pos", msg)
    time.sleep(0.5)
    new_state = b.recv_msg("franka_state", -1)
    print('next pos: {}'.format(new_state.get_c_pos()))
    counter += 1
