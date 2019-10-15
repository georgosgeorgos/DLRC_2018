import numpy as np
import py_at_broker as pab

# import SLRobot
import time
import signal
import sys
import time
import sys
import os
import subprocess
import _pickle as pickle
import threading

# from matplotlib import pyplot as plt
# import pudb

import importlib
import yaml
import tensorflow as tf
from autemp.base import Experiment, Tracker, Roller
from autemp.franka import FrankaConfig
from autemp.data import DataUtils

import pdb
import argparse


REGULAR = True


class ControlLoop:
    def __init__(self, broker_id):
        """
        Init control loop
        :param pars:
        """

        # channels and messages
        self.broker_server_id = broker_id
        self.image_channel_id = FrankaConfig.connections.image_channel_id
        self.state_channel_id = FrankaConfig.connections.state_channel_id
        self.lidar_channel_id = FrankaConfig.connections.lidar_channel_id

        if REGULAR:
            self.image_msg_type = pab.MsgType.franka_images
            self.state_msg_type = pab.MsgType.franka_state
            self.lidar_msg_type = pab.MsgType.franka_images
        else:
            self.image_msg_type = pab.MsgType.NumpyMsg
            self.state_msg_type = pab.MsgType.NumpyMsg
            self.lidar_msg_type = pab.MsgType.NumpyMsg

        self.image_pred_channel_id = FrankaConfig.connections.image_pred_channel_id
        self.image_pred_msg_type = pab.MsgType.NumpyMsg
        self.state_pred_channel_id = FrankaConfig.connections.state_pred_channel_id
        self.state_pred_msg_type = pab.MsgType.NumpyMsg
        self.lidar_pred_channel_id = FrankaConfig.connections.lidar_pred_channel_id
        self.lidar_pred_msg_type = pab.MsgType.NumpyMsg

        n_joints = FrankaConfig.robot.n_joints
        self.n_joints = n_joints

        # shapes
        self.state_robot_shape = [2 * n_joints]
        self.endeff_robot_shape = [2 * FrankaConfig.data.endeff_size]
        self.image_shape = FrankaConfig.data.img_shape
        self.lidar_shape = [FrankaConfig.data.lidar_num]

        # obs ints
        n_lidar = FrankaConfig.data.lidar_num
        n_image = np.prod(FrankaConfig.data.img_shape)
        self.state_int = np.arange(0, 2 * n_joints)
        self.lidar_int = np.arange(2 * n_joints, 2 * n_joints + n_lidar)
        self.image_int = np.arange(2 * n_joints + n_lidar, 2 * n_joints + n_lidar + n_image)

        # quantities to keep track of
        self.image = np.zeros((1, FrankaConfig.data.img_size))
        self.image_time = 0.0
        self.image_recv_time = 0.0
        self.image_fnumber = 0

        self.endeff_x = np.zeros((1, FrankaConfig.robot.n_endeff))
        self.endeff_xd = np.zeros((1, FrankaConfig.robot.n_endeff))
        self.state_q = np.zeros((1, FrankaConfig.robot.n_joints))
        self.state_qd = np.zeros((1, FrankaConfig.robot.n_joints))
        self.state_time = 0.0
        self.state_recv_time = 0.0
        self.state_fnumber = 0

        self.lidar = np.zeros((1, FrankaConfig.data.lidar_num))
        self.lidar_time = 0.0
        self.lidar_recv_time = 0.0
        self.lidar_fnumber = 0

    def connect_to_channels(self):
        # try to connect to communication channels

        # os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(35))
        self.broker_server = pab.broker(self.broker_server_id)

        self.broker_server.request_signal(self.state_channel_id, self.state_msg_type)
        # self.broker_server.request_signal(self.image_channel_id, self.image_msg_type)
        # self.broker_server.request_signal(self.lidar_channel_id, self.lidar_msg_type)

        self.broker_server.register_signal(self.image_pred_channel_id, self.image_pred_msg_type)
        self.broker_server.register_signal(self.state_pred_channel_id, self.state_pred_msg_type)
        self.broker_server.register_signal(self.lidar_pred_channel_id, self.lidar_pred_msg_type)

    def load_model(self, path_exp_yaml, path_model_ckpt, reload=True):
        """
        Load model
        :param exp_dir:
        :param exp_name:
        :return:
        """

        with open(path_exp_yaml) as fid_yaml:
            pars = yaml.load(fid_yaml)
        self.model_pars = pars

        data_def = pars["data"]
        data_pars = dict()
        data_pars["lidar_num"] = len(data_def["mask_lidar"])
        data_pars["img_shape"] = [len(data_def["mask_image"])] + data_def["image_shape"]

        #   Define model
        module_autemp = importlib.import_module("autemp.base")
        module_dvbf = importlib.import_module("lauge.sgvb.dvbf")

        Likelihood = getattr(module_autemp, pars["model"]["likelihood"]["name"])
        Loss_control_z = getattr(module_autemp, pars["model"]["fn_control_z_loss"]["name"])
        Loss_control_x = getattr(module_autemp, pars["model"]["fn_control_x_loss"]["name"])
        Transition = getattr(module_dvbf, pars["model"]["transition"]["name"])
        if False:
            Update = getattr(module_dvbf, pars["model"]["encoder"]["name"])
        else:
            Update = getattr(module_autemp, pars["model"]["encoder"]["name"])

        Initial = getattr(module_autemp, pars["model"]["init"]["name"])
        GDVBF = getattr(module_dvbf, "GenericDeepVariationalBayesFilter")

        class DeepVariationalBayesFilter(GDVBF, Transition, Initial, Likelihood, Update):
            pass

        np.random.seed(pars["inference"]["seed_np"])
        tf.set_random_seed(pars["inference"]["seed_tf"])
        # tf.reset_default_graph()
        # with tf.device('/gpu:0'):
        # define the model
        self.model = DeepVariationalBayesFilter(
            use_placeholder=True,
            seq_length=pars["inference"]["n_timesteps"],
            n_obs=pars["model"]["n_obs"],
            n_control=pars["model"]["n_control"],
            n_state=pars["model"]["n_state"],
            n_alpha=pars["model"]["n_alpha"],
            n_hiddens_lik=pars["model"]["likelihood"]["n_hiddens"],
            transfers_lik=pars["model"]["likelihood"]["transfers"],
            n_hiddens_enc=pars["model"]["encoder"]["n_hiddens"],
            transfers_enc=pars["model"]["encoder"]["transfers"],
            n_hiddens_init=pars["model"]["init"]["n_hiddens"],
            transfers_init=pars["model"]["init"]["transfers"],
            n_hiddens_zerothtransition=pars["model"]["transition_init"]["n_hiddens"],
            transfers_zerothtransition=pars["model"]["transition_init"]["transfers"],
            n_hiddens_gen=pars["model"]["transition"]["n_hiddens"],
            transfers_gen=pars["model"]["transition"]["transfers"],
            n_hiddens_policy=pars["model"]["policy"]["n_hiddens"],
            transfers_policy=pars["model"]["policy"]["transfers"],
            n_hiddens_planning=pars["model"]["planning"]["n_hiddens"],
            transfers_planning=pars["model"]["planning"]["transfers"],
            zeroth_transition=True,
            n_initialobs=pars["model"]["n_initialobs"],
            uncertain_weights=True,
            batch_size=pars["inference"]["n_batch"],
            beta_T=pars["inference"]["beta_T"],
            control=pars["inference"]["control_on"],
            old_policy_prior=pars["inference"]["old_policy_prior"],
            max_control=pars["inference"]["control_max"],  # self.data["control_max"],
            n_steps_emp=pars["inference"]["n_steps_emp"],
            fn_control_z_loss=Loss_control_z,
            fn_control_x_loss=Loss_control_x,
            epoch_start=pars["state"]["n_epoch"],
            data_pars=data_pars,
        )

        # define a predictor to do the evaluations
        self.predictor = Tracker(self.model, self.model.policy_from_obs)
        self.roller = Roller(self.model, self.model.policy_from_obs)

        # load model
        self.saver = tf.train.Saver()
        if os.path.exists(path_model_ckpt + ".meta"):
            self.saver.restore(self.model.sess, path_model_ckpt)
            print("Model %s loaded!" % (path_model_ckpt))

        # polic
        self.path_exp_yaml = path_exp_yaml
        self.path_model_ckpt = path_model_ckpt
        self.counter_current_model = 0
        self.time_model_update = time.clock_gettime(time.CLOCK_MONOTONIC)
        self.time_model_update_check = time.clock_gettime(time.CLOCK_MONOTONIC)
        self.counter_burnin = 0

    def load_model_dummy(self):
        class DummyPredictor:
            def __init__(self, model):
                self.model = model

            def initialize(self, data):
                pass

            def step(self, data):
                return None

        # load model
        self.model = None
        self.predictor = DummyPredictor(self.model)

        self.saver = None

        self.path_exp_yaml = None
        self.path_model_ckpt = None
        self.counter_current_model = 0
        self.time_model_update = time.clock_gettime(time.CLOCK_MONOTONIC)
        self.time_model_update_check = time.clock_gettime(time.CLOCK_MONOTONIC)
        self.counter_burnin = 0

    def rollout_and_send(self, z0_mean, z0_var):
        """
        rollout paths and send message

        """
        z_paths_mean, u_paths_mean, x_paths_mean, z_paths_var, u_paths_var, x_paths_var = self.roller.evaluate_model_and_obs_policy_from_latent(
            z0_mean, z0_var, self.t_rollout, self.n_samples_rollout
        )

        print(x_paths_mean.shape)

        # msg_image = pab.fb_msg(pab.MsgType.NumpyMsg, np.concat(x_paths_mean[:, self.image_int], x_paths_var[:, self.image_int], axis=1))
        msg_state = pab.fb_msg(
            pab.MsgType.NumpyMsg,
            np.concatenate([x_paths_mean[:, self.state_int], x_paths_var[:, self.state_int]], axis=1),
        )
        # msg_lidar = pab.fb_msg(pab.MsgType.NumpyMsg, np.concat(x_paths_mean[:, self.lidar_int], x_paths_var[:, self.lidar_int], axis=1))

        print("Sending rollout")

        # self.broker_server.send_msg(self.image_pred_channel_id, msg_image)
        self.broker_server.send_msg(self.state_pred_channel_id, msg_state)
        # self.broker_server.send_msg(self.lidar_pred_channel_id, msg_lidar)

    def run(self, dt_rollout=3.0, n_samples_rollout=10, t_rollout=30):
        """
        Run control loop
        :return:
        """

        INTERACT = True

        self.counter_current_model = 0
        self.counter_life = 0

        self.t_rollout = t_rollout
        self.dt_rollout = dt_rollout
        self.n_samples_rollout = n_samples_rollout

        # some stuff for control
        INIT_ON = True
        self.dc_burnin = 1
        self.counter_burnin = 0
        self.wtf_init = np.zeros(
            (
                self.model.n_initialobs,
                2 * FrankaConfig.robot.n_joints + FrankaConfig.data.lidar_num + FrankaConfig.data.img_size,
            )
        )

        #  some stuff for masking data / old experiments
        mask_state = self.model_pars["data"]["mask_state"]

        counter = 0

        default_zero_img = np.zeros((1, FrankaConfig.data.img_size))
        default_zero_lidar = np.zeros((1, FrankaConfig.data.lidar_num))

        while True:
            # update counters
            self.counter_current_model += 1
            self.counter_life += 1

            counter += 1

            # if not INTERACT:
            #    time.sleep(0.5)
            #    print((counter,time.clock_gettime(time.CLOCK_MONOTONIC)))

            if REGULAR:
                # recv_start            = time.clock_gettime(time.CLOCK_MONOTONIC)
                #
                # msg_image             = self.broker_server.recv_msg(self.image_channel_id, -1)
                # self.image            = msg_image.get_data().reshape((1, -1))
                # self.image_time       = time.clock_gettime(time.CLOCK_MONOTONIC)
                # self.image_recv_time  = msg_image.get_timestamp()
                # self.image_fnumber    = msg_image.get_fnumber()
                # recv_stop             = time.clock_gettime(time.CLOCK_MONOTONIC)

                time.sleep(0.5)
                self.image = default_zero_img

                msg_state = self.broker_server.recv_msg(self.state_channel_id, -1)

                self.endeff_x = msg_state.get_c_pos().reshape((1, -1))
                self.endeff_xd = msg_state.get_c_vel().reshape((1, -1))
                self.state_q = msg_state.get_j_pos().reshape((1, -1))
                self.state_qd = msg_state.get_j_vel().reshape((1, -1))
                self.state_time = msg_state.get_timestamp()
                self.state_recv_time = msg_state.get_timestamp()
                self.state_fnumber = msg_state.get_fnumber()

                # msg_lidar            = self.broker_server.recv_msg(self.lidar_channel_id, -1)
                # self.lidar           = msg_lidar.get_data().reshape((1, -1))
                # self.lidar_time      = time.clock_gettime(time.CLOCK_MONOTONIC)
                # self.lidar_recv_time = msg_lidar.get_timestamp()
                # self.lidar_fnumber   = msg_lidar.get_fnumber()
                self.lidar = default_zero_lidar

                print(counter)

            else:

                # getting numpy messages
                msg_image = self.broker_server.recv_msg(self.image_channel_id, -1)
                msg_state = self.broker_server.recv_msg(self.state_channel_id)
                msg_lidar = self.broker_server.recv_msg(self.lidar_channel_id)

                msg_state_data = msg_state.get()

                # processing numpy messages
                self.image = msg_image.get().reshape((-1, FrankaConfig.data.img_size))
                self.image_time = time.clock_gettime(time.CLOCK_MONOTONIC)
                self.image_recv_time = self.image_time
                self.image_fnumber = counter

                self.endeff_x = np.zeros((1, FrankaConfig.robot.n_endeff))
                self.endeff_xd = np.zeros((1, FrankaConfig.robot.n_endeff))
                self.state_q = msg_state_data[:, : FrankaConfig.robot.n_joints].reshape(
                    (-1, FrankaConfig.robot.n_joints)
                )
                self.state_qd = msg_state_data[:, FrankaConfig.robot.n_joints :].reshape(
                    (-1, FrankaConfig.robot.n_joints)
                )
                self.state_time = time.clock_gettime(time.CLOCK_MONOTONIC)
                self.state_recv_time = self.state_time
                self.state_fnumber = counter

                msg_lidar_data = msg_lidar.get().reshape((-1, FrankaConfig.data.lidar_num))
                self.lidar = msg_lidar_data
                self.lidar_time = time.clock_gettime(time.CLOCK_MONOTONIC)
                self.lidar_recv_time = self.lidar_time
                self.lidar_fnumber = counter

            wtf_state = np.concatenate((self.state_q, self.state_qd, self.lidar, self.image), axis=1)

            if INIT_ON:

                if True:
                    # if in burnin store data for initialisation
                    if (self.counter_current_model % self.dc_burnin == 0) and (
                        self.counter_burnin < self.model.n_initialobs
                    ):
                        self.wtf_init[self.counter_burnin, :] = wtf_state
                        self.counter_burnin += 1
                        print("Data recorded for init.")
                    # if enough data do initialisation
                    if self.counter_burnin == self.model.n_initialobs:
                        INIT_ON = False

                        if True:
                            self.predictor.initialize(self.wtf_init[:, : 2 * self.n_joints])
                            t_last_rollout = time.time()
                            print("Model initilised.")
            else:
                if True:
                    # t = time.time()
                    tmp = wtf_state.reshape((1, -1))
                    z_pred_mean, z_pred_var, u_pred_mean, u_pred_var, x_pred_mean, x_pred_var = self.predictor.step(
                        tmp[(0), : 2 * self.n_joints].reshape((1, -1))
                    )

                    if time.time() - t_last_rollout > dt_rollout:
                        t_last_rollout = time.time()
                        thread_roll = threading.Thread(target=self.rollout_and_send, args=(z_pred_mean, z_pred_var))
                        thread_roll.start()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", help="ip")
    parser.add_argument("-d", help="[string], directory to load model ckpt from experiment")
    parser.add_argument("-t", type=float, help="[float], dt for rollout")
    parser.add_argument("-l", type=int, help="[int], length of rollouts")
    parser.add_argument("-s", type=int, help="[int], samples for rollout")
    args = parser.parse_args()

    broker_ip = args.i
    dir_model = args.d
    dt_rollout = float(args.t)
    t_rollout = int(args.l)
    n_samples_rollout = int(args.s)

    if broker_ip is None:
        broker_ip = FrankaConfig.connections.broker_server_id
    else:
        broker_ip = str(broker_ip)

    control_loop = ControlLoop(broker_ip)

    if True:
        model_name = os.path.split(dir_model)[-1]
        path_model_ckpt = os.path.join(dir_model, model_name)
        path_model_yaml = os.path.join(dir_model, model_name + ".yaml")

        control_loop.load_model(path_model_yaml, path_model_ckpt)
    else:
        control_loop.load_model_dummy()

    control_loop.connect_to_channels()
    control_loop.run(dt_rollout=dt_rollout, n_samples_rollout=n_samples_rollout, t_rollout=t_rollout)
