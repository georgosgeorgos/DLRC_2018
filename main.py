import py_at_broker as pab
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns

b = pab.broker()
print(b.register_signal("franka_target_pos", pab.MsgType.target_pos))
time.sleep(0.5)
print(b.request_signal("franka_lidar", pab.MsgType.franka_lidar))
print(b.request_signal("franka_state", pab.MsgType.franka_state))
time.sleep(0.5)

counter = 0
default_pos_c = np.array([0.62, 0.00, 0.56])

# def default_pos():
#     pos_c = np.array([ 6.23963393e-01, -5.11126213e-06,  5.59791033e-01])
#     msg = pab.target_pos_msg()
#     msg.set_ctrl_t(pab.CtrlType.Cartesian)
#     msg.set_pos(pos_c)
#     msg.set_timestamp(time.clock_gettime(time.CLOCK_MONOTONIC))
#     msg.set_fnumber(counter)
#     msg.set_time_to_go(2.)
#     b.send_msg("franka_target_pos", msg)
#     return None

def f_msg(pos):
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

n_points = 10
n_runs = 2
time2go = 1.
delta = 2 * np.pi / n_points
R = 0.2

lidar_list = []
c = 0
n_samples=100

while True:
    f_msg(default_pos_c)
    time.sleep(0.5)
    state = b.recv_msg("franka_state", -1)    
    current_pos_c = state.get_c_pos()
    print("default pos: {}".format(default_pos_c))
    print('current pos: {}'.format(current_pos_c))
    counter += 1
    time.sleep(2.0)
    for n in range(1, n_runs+1):
        print("RUN:", n)
        print()
        for theta in np.arange(0, 2*np.pi, delta):

            # define next position
            pos_c = default_pos_c + [R*np.sin(theta), R*np.cos(theta), 0]

            # move
            print("move robot")
            f_msg(pos_c)
            # stop
            #time.sleep(0.5)
            print("start lidar data collected")
            # measure
            c = 0
            while c < n_samples:
                lidar = b.recv_msg("franka_lidar", -1)
                lidar_list.append(lidar.get_data())
                counter += 1
                c += 1

            #lidar_array = np.array(lidar_list)
            print("end lidar data collected")
            #time.sleep(0.5)
            #lidar_array[lidar_array > 1900] = 0
            #print(lidar_array.shape)

            # print("plot hist")
            # for j in range(9):
            #     plt.plot(lidar_array[:,j])
            #     plt.show()
            #     plt.hist(lidar_array[:,j][lidar_array[:,j] > 0], bins=30, density=True)
            #     plt.show()

            # plt.show()

            new_state = b.recv_msg("franka_state", -1)
            print('next pos: {}'.format(new_state.get_c_pos()))
            counter += 1

    lidar_array = np.array(lidar_list)
    print(lidar_array.shape)
    np.save("lidar_data.npy", lidar_array)
    break



