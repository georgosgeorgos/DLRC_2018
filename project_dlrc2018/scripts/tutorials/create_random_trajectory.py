import time

import numpy as np
import py_at_broker as pab

b = pab.broker()
print(b.register_signal("franka_target_pos", pab.MsgType.target_pos))
time.sleep(0.5)
print(b.request_signal("franka_lidar", pab.MsgType.franka_lidar))
print(b.request_signal("realsense_images", pab.MsgType.realsense_image))
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
print("initial pos: {}".format(current_pos_c))

n_points = 100
n_runs = 10
time2go = 1.0
delta = 2 * np.pi / n_points
R = 0.2

lidar_list = []
trajectory = []
c = 0
n_samples = 1


WallMin = [0.28, -0.78, 0.02]
WallMax = [0.82, 0.78, 1.08]

X = np.random.uniform(WallMin[0], WallMax[0], 100)
Y = np.random.uniform(WallMin[1], WallMax[1], 100)
Z = np.random.uniform(WallMin[2], WallMax[2], 100)

runs = 0

while runs < 10:
    print("RUNS: ", runs)
    try:
        pos_msg(default_pos_c)
        time.sleep(2.0)
        state = b.recv_msg("franka_state", -1)
        pos_c = state.get_c_pos()
        print("default pos: {}".format(default_pos_c))
        print("current pos: {}".format(pos_c))
        counter += 1
        time.sleep(2.0)
        for _ in range(300):
            # define next position
            pos_c = pos_c + np.random.uniform(-0.02, 0.02, 3)
            # [x, y, z]
            print(pos_c)
            trajectory.append(pos_c)
            # move
            print("move robot")
            pos_msg(pos_c)
            # stop
            # time.sleep(0.5)
            print("start lidar data collected")
            # measure
            lidar = b.recv_msg("franka_lidar", -1)
            lidar_list.append(lidar.get_data())
            counter += 1

            new_state = b.recv_msg("franka_state", -1)
            print("next pos: {}".format(new_state.get_c_pos()))
            counter += 1

        lidar_array = np.array(lidar_list)
        trajectory_array = np.array(trajectory)
        np.save("./random_data/lidar_" + str(runs) + ".npy", lidar_array)
        np.save("./random_data/trajectory" + str(runs) + ".npy", trajectory_array)
        runs += 1
    except KeyboardInterrupt:
        lidar_array = np.array(lidar_list)
        trajectory_array = np.array(trajectory)
        np.save("./random_data/lidar_" + str(runs) + ".npy", lidar_array)
        np.save("./random_data/trajectory" + str(runs) + ".npy", trajectory_array)
        runs += 1
