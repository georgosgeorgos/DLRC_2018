import numpy as np
import py_at_broker as pab
import time
import signal
import sys
import time
import sys
import os
import subprocess
import _pickle as pickle

from autemp.franka import FrankaConfig
import argparse


import pdb


class RecordingLoop:
    def __init__(self):
        """
        Init control loop
        :param pars:
        """

        # channels and messages
        self.broker_server_id = FrankaConfig.connections.broker_server_id
        self.image_channel_id = FrankaConfig.connections.image_channel_id
        self.image_msg_type = pab.MsgType.franka_images
        self.state_channel_id = FrankaConfig.connections.state_channel_id
        self.state_msg_type = pab.MsgType.franka_state
        self.control_channel_id = FrankaConfig.connections.control_channel_id
        self.control_msg_type = pab.MsgType.des_tau

        n_joints = FrankaConfig.robot.n_joints
        self.n_joints = n_joints

        # shapes
        self.state_robot_shape = [2 * n_joints]
        self.endeff_robot_shape = [2 * FrankaConfig.data.endeff_size]
        self.control_shape = [n_joints]
        self.state_img_shape = FrankaConfig.data.img_shape

        # state we keep track of
        self.state_img = np.zeros(self.state_img_shape)
        self.state_q = np.zeros(n_joints)
        self.state_qd = np.zeros(n_joints)
        self.endeff_x = np.zeros(FrankaConfig.data.endeff_size)
        self.endeff_xd = np.zeros(FrankaConfig.data.endeff_size)

        self.img_time = 0.0
        self.state_time = 0.0
        self.control_time = 0.0

        self.img_recv_time = 0.0
        self.state_recv_time = 0.0
        self.control_recv_time = 0.0

        self.state_fnumber = 0
        self.img_fnumber = 0
        self.control_fnumber = 0

        # parameters for data storage
        self.recording_dir = FrankaConfig.recording.recording_dir
        self.recording_tag = FrankaConfig.recording.recording_tag
        self.recording_serial_const = FrankaConfig.recording.recording_serial_const
        self.recording_serial_start = FrankaConfig.recording.reccording_serial_start

        # recording buffer
        self.RECORDING_ON = False
        self.buffer_size = FrankaConfig.recording.recording_buffer_size + 1  # the +1 is for the control shifting
        self.buffer_serial = 0
        self.buffer_counter = 0
        self.buffer_data = {
            "img": np.zeros([self.buffer_size] + [FrankaConfig.data.img_size]),
            "img_time": np.zeros([self.buffer_size, 1]),
            "img_recv_time": np.zeros([self.buffer_size, 1]),
            "img_fnumber": np.zeros([self.buffer_size, 1]),
            "state": np.zeros([self.buffer_size] + self.state_robot_shape),
            "endeff": np.zeros([self.buffer_size] + self.endeff_robot_shape),
            "state_time": np.zeros([self.buffer_size, 1]),
            "state_recv_time": np.zeros([self.buffer_size, 1]),
            "state_fnumber": np.zeros([self.buffer_size, 1]),
            "control": np.zeros([self.buffer_size] + self.control_shape),
            "control_time": np.zeros([self.buffer_size, 1]),
            "control_recv_time": np.zeros([self.buffer_size, 1]),
            "control_fnumber": np.zeros([self.buffer_size, 1]),
        }

        # Felix's counter
        self.FIRST_TIME = True

        # some counters
        self.counter_life = 0
        self.counter_current_model = 0
        self.path_current_model = ""
        self.time_model_update = 0

    def connect_to_channels(self, IMG_ON=False):
        # try to connect to communication channels

        # os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(35))
        self.broker_server = pab.broker(self.broker_server_id)
        self.broker_server.request_signal(self.state_channel_id, self.state_msg_type)

        if IMG_ON:
            self.broker_server.request_signal(self.image_channel_id, self.image_msg_type)

        if True:
            print("Connection setup worked!")

    def run(self, IMG_ON=False, SYNC_ON=False, name_recording="i_should_not_exist", recording_serial=0):
        """
        Run control loop
        :return:
        """

        dir_experiment = os.path.join(self.recording_dir, name_recording)

        if not os.path.exists(dir_experiment):
            os.makedirs(dir_experiment)

        dir_data = os.path.join(dir_experiment, "data")
        if not os.path.exists(dir_data):
            os.makedirs(dir_data)

        dir_models = os.path.join(dir_experiment, "models")
        if not os.path.exists(dir_models):
            os.makedirs(dir_models)

        dir_configs = os.path.join(dir_experiment, "configs")
        if not os.path.exists(dir_configs):
            os.makedirs(dir_configs)

        self.buffer_serial = 0
        self.buffer_counter = 0

        counter = 0

        while True:

            counter += 1

            # if not INTERACT:
            #    time.sleep(0.5)
            #    print((counter,time.clock_gettime(time.CLOCK_MONOTONIC)))

            if True:
                #
                # Request state message
                #
                recv_start = time.clock_gettime(time.CLOCK_MONOTONIC)

                msg_state = self.broker_server.recv_msg(self.state_channel_id, -1)

                recv_stop = time.clock_gettime(time.CLOCK_MONOTONIC)

                self.state_time = recv_stop

                # trig_time = time.clock_gettime(time.CLOCK_MONOTONIC)
                # Catch no message for long time
                if (recv_stop - recv_start) > 0.2:
                    print("Reset target")
                    firsttime = 0

                # Extract message A
                q_A = msg_state.get_j_pos()
                qd_A = msg_state.get_j_vel()
                endeff_x = msg_state.get_c_pos()
                endeff_xd = msg_state.get_c_vel()
                control_prev = msg_state.get_last_cmd()
                frame_counter = msg_state.get_fnumber()
                recv_time = msg_state.get_timestamp()

            else:
                #
                #   WARNING: this is for debugging/testing purposes only
                #

                q_A = np.ones(FrankaConfig.robot.n_joints)
                qd_A = np.ones(FrankaConfig.robot.n_joints)
                endeff_x = np.ones(FrankaConfig.robot.n_endeff)
                endeff_xd = np.ones(FrankaConfig.robot.n_endeff)
                control_prev = np.ones(FrankaConfig.robot.n_joints)
                frame_counter = counter
                recv_time = time.clock_gettime(time.CLOCK_MONOTONIC)

            self.endeff_x = endeff_x
            self.endeff_xd = endeff_xd
            self.state_q = q_A
            self.state_qd = qd_A
            self.state_time = recv_time
            self.state_recv_time = recv_time
            self.state_fnumber = frame_counter

            self.control_prev = control_prev
            self.control_prev_time = self.state_time
            self.control_prev_recv_time = self.state_recv_time
            self.control_prev_fnumber = self.state_fnumber

            #
            # Request image message
            #
            if IMG_ON:
                msg_image = self.broker_server.recv_msg(self.image_channel_id, -1)
                self.state_img = msg_image.get_data()
                self.img_time = time.clock_gettime(time.CLOCK_MONOTONIC)
                self.img_recv_time = msg_image.get_timestamp()
                self.img_fnumber = msg_image.get_fnumber()

            else:
                self.state_img = np.zeros(self.state_img_shape)
                self.img_time = time.clock_gettime(time.CLOCK_MONOTONIC)
                self.img_recv_time = self.img_time
                self.img_fnumber = counter
            #
            # data processing starts here
            #

            if self.buffer_counter < self.buffer_size:
                # add data to buffer

                # pdb.set_trace()

                self.buffer_data["img"][self.buffer_counter, :] = np.ravel(self.state_img)
                self.buffer_data["img_time"][self.buffer_counter, :] = self.img_time
                self.buffer_data["img_recv_time"][self.buffer_counter, :] = self.img_recv_time
                self.buffer_data["img_fnumber"][self.buffer_counter, :] = self.img_fnumber
                self.buffer_data["state"][self.buffer_counter, :] = np.concatenate((self.state_q, self.state_qd))
                self.buffer_data["endeff"][self.buffer_counter, :] = np.concatenate((self.endeff_x, self.endeff_xd))
                self.buffer_data["state_time"][self.buffer_counter, :] = self.state_time
                self.buffer_data["state_recv_time"][self.buffer_counter, :] = self.state_recv_time
                self.buffer_data["state_fnumber"][self.buffer_counter, :] = self.state_fnumber
                self.buffer_data["control"][self.buffer_counter, :] = self.control_prev  # self.control_out
                self.buffer_data["control_time"][self.buffer_counter, :] = self.control_prev_time  # self.control_time
                self.buffer_data["control_recv_time"][
                    self.buffer_counter, :
                ] = self.control_prev_recv_time  # self.control_time
                self.buffer_data["control_fnumber"][
                    self.buffer_counter, :
                ] = self.control_prev_fnumber  # self.control_time

                self.buffer_counter += 1

                # print("Data recorded for saving:%d" % (self.buffer_counter))

            else:
                # serialise buffer

                # if False:
                #     fname = self.recording_tag + "_" + str(self.recording_serial_const) \
                #             + "_" + str(self.recording_serial_start + self.buffer_serial) + ".tfrecords"
                #     fname = os.path.join(self.recording_dir, fname)
                #
                #     if os.path.exists(fname):
                #         raise Exception("Data file %s already exists!" % (fname))
                #
                #     DataUtils.dict_to_tfrecords(self.buffer_data, fname, FrankaConfig.data.K_subsample)
                # else:
                fname = (
                    name_recording
                    + "_"
                    + self.recording_tag
                    + "_"
                    + str(recording_serial).zfill(4)
                    + "_"
                    + str(self.buffer_serial).zfill(4)
                    + ".raw"
                )
                fname = os.path.join(dir_data, fname)

                if os.path.exists(fname):
                    raise Exception("Data file %s already exists!" % (fname))

                with open(fname, "wb") as fstream:
                    pickle.dump(self.buffer_data, fstream)

                if SYNC_ON:
                    # lauch data processor and syncer
                    subprocess.Popen(["python", "process_rec_and_sync.py", " -f ", fname, " -s ", str(int(SYNC_ON))])

                # move on
                self.buffer_counter = 0
                self.buffer_serial += 1

                print("Recording to file %s" % (fname))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", help="[string], experiment name for saving data, same as above for cont training")
    parser.add_argument("-i", type=int, help="[int] (as bool), image 1/0")
    parser.add_argument("-s", type=int, help="[int] sync with training on")
    parser.add_argument("-n", type=int, help="[int] experiment serial to add in front of sequence numbers ")
    args = parser.parse_args()

    name_recording = args.d
    IMG_ON = bool(args.i)
    SYNC_ON = bool(args.s)
    recording_serial = int(args.n)

    recording_loop = RecordingLoop()

    recording_loop.connect_to_channels(IMG_ON)

    recording_loop.run(IMG_ON=IMG_ON, SYNC_ON=SYNC_ON, name_recording=name_recording, recording_serial=recording_serial)
