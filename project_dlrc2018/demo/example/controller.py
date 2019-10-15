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
# from matplotlib import pyplot as plt
# import pudb

import importlib
import yaml
import tensorflow as tf
from autemp.base import Experiment, Predictor, Roller
from autemp.franka import FrankaConfig
from autemp.data import DataUtils

import pdb
import argparse

class ControlLoop():
    def __init__(self):
        """
        Init control loop
        :param pars:
        """

        # channels and messages
        self.broker_server_id   = FrankaConfig.connections.broker_server_id
        self.image_channel_id   = FrankaConfig.connections.image_channel_id
        self.image_msg_type     = pab.MsgType.franka_images
        self.state_channel_id   = FrankaConfig.connections.state_channel_id
        self.state_msg_type     = pab.MsgType.franka_state
        self.lidar_channel_id   = FrankaConfig.connections.lidar_channel_id
        self.lidar_msg_type     = pab.MsgType.franka_images

        n_joints      = FrankaConfig.robot.n_joints
        self.n_joints = n_joints

        # shapes
        self.state_robot_shape  = [2*n_joints]
        self.endeff_robot_shape = [2*FrankaConfig.data.endeff_size]
        self.policy_shape       = [2*n_joints]
        self.control_shape      = [n_joints]
        self.state_img_shape    = FrankaConfig.data.img_shape

        # state we keep track of
        self.state_img    = np.zeros(self.state_img_shape)
        self.state_q      = np.zeros(n_joints)
        self.state_qd     = np.zeros(n_joints)
        self.endeff_x     = np.zeros(FrankaConfig.data.endeff_size)
        self.endeff_xd    = np.zeros(FrankaConfig.data.endeff_size)
        self.policy_mean  = np.zeros(n_joints)
        self.policy_var   = np.zeros(n_joints)
        self.control_out  = np.zeros(n_joints)

        self.img_time     = 0.0
        self.state_time   = 0.0
        self.control_time = 0.0

        self.img_recv_time     = 0.0
        self.state_recv_time   = 0.0
        self.control_recv_time = 0.0

        self.state_fnumber   = 0
        self.img_fnumber     = 0
        self.control_fnumber = 0

        # parameters for data storage
        self.recording_dir  = FrankaConfig.recording.recording_dir
        self.recording_tag  = FrankaConfig.recording.recording_tag
        self.recording_serial_const = FrankaConfig.recording.recording_serial_const
        self.recording_serial_start = FrankaConfig.recording.reccording_serial_start

        # predictor
        self.predictor = None

        # Felix's counter
        self.FIRST_TIME = True

        # some counters
        self.counter_life          = 0
        self.counter_current_model = 0
        self.path_current_model    = ""
        self.time_model_update     = 0

        # update times
        self.dt_model_update  = FrankaConfig.updates.dt_model_update
        self.dc_policy_update = FrankaConfig.updates.dc_policy_update
        self.dc_burnin        = FrankaConfig.updates.dc_burnin

        self.counter_burnin   = 0

    def connect_to_channels(self):
        # try to connect to communication channels

        #os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(35))
        self.broker_server = pab.broker(self.broker_server_id)
        self.broker_server.request_signal(self.state_channel_id, self.state_msg_type)
        self.broker_server.request_signal(self.image_channel_id, self.image_msg_type)
        self.broker_server.request_signal(self.lidar_channel_id, self.lidar_msg_type)


    def load_model(self, path_exp_yaml, path_model_ckpt, reload = False):
        """
        Load model
        :param exp_dir:
        :param exp_name:
        :return:
        """

        with open(path_exp_yaml) as fid_yaml:
            pars = yaml.load(fid_yaml)
        self.model_pars = pars

        #   Define model
        module_autemp = importlib.import_module("autemp.base")
        module_dvbf   = importlib.import_module("lauge.sgvb.dvbf")

        Likelihood     = getattr(module_autemp, self.pars["model"]["likelihood"]["name"])
        Loss_control_z = getattr(module_autemp, self.pars["model"]["fn_control_z_loss"]["name"])
        Loss_control_x = getattr(module_autemp, self.pars["model"]["fn_control_x_loss"]["name"])
        Transition = getattr(module_dvbf, self.pars["model"]["transition"]["name"])
        if False:
            Update  = getattr(module_dvbf, self.pars["model"]["encoder"]["name"])
        else:
            Update = getattr(module_autemp, self.pars["model"]["encoder"]["name"])

        Initial    = getattr(module_dvbf, self.pars["model"]["init"]["name"])
        GDVBF      = getattr(module_dvbf, "GenericDeepVariationalBayesFilter")

        class DeepVariationalBayesFilter(GDVBF, Transition, Initial, Likelihood, Update):
            pass

        np.random.seed(self.pars["inference"]["seed_np"])
        tf.set_random_seed(self.pars["inference"]["seed_tf"])
        # tf.reset_default_graph()
        #with tf.device('/gpu:0'):
        # define the model
        self.model = DeepVariationalBayesFilter(n_obs = self.pars["model"]["n_obs"],
                                                n_control = self.pars["model"]["n_control"],
                                                n_state = self.pars["model"]["n_state"],
                                                n_alpha = self.pars["model"]["n_alpha"],
                                                n_samples = self.data["n_samples_train"],
                                                n_hiddens_lik = self.pars["model"]["likelihood"]["n_hiddens"],
                                                transfers_lik = self.pars["model"]["likelihood"]["transfers"],
                                                n_hiddens_enc = self.pars["model"]["encoder"]["n_hiddens"],
                                                transfers_enc = self.pars["model"]["encoder"]["transfers"],
                                                n_hiddens_init = self.pars["model"]["init"]["n_hiddens"],
                                                transfers_init = self.pars["model"]["init"]["transfers"],
                                                n_hiddens_zerothtransition = self.pars["model"]["transition_init"]["n_hiddens"],
                                                transfers_zerothtransition = self.pars["model"]["transition_init"]["transfers"],
                                                n_hiddens_gen = self.pars["model"]["transition"]["n_hiddens"],
                                                transfers_gen = self.pars["model"]["transition"]["transfers"],
                                                n_hiddens_policy = self.pars["model"]["policy"]["n_hiddens"],
                                                transfers_policy = self.pars["model"]["policy"]["transfers"],
                                                n_hiddens_planning = self.pars["model"]["planning"]["n_hiddens"],
                                                transfers_planning = self.pars["model"]["planning"]["transfers"],
                                                zeroth_transition = True,
                                                n_initialobs = self.pars["model"]["n_initialobs"],
                                                uncertain_weights = True,
                                                batch_size = self.pars["inference"]["n_batch"],
                                                beta_T = self.pars["inference"]["beta_T"],
                                                control = self.pars["inference"]["control_on"],
                                                old_policy_prior = self.pars["inference"]["old_policy_prior"],
                                                max_control = self.data["control_max"],
                                                n_steps_emp = self.pars["inference"]["n_steps_emp"],
                                                fn_control_z_loss = Loss_control_z,
                                                fn_control_x_loss = Loss_control_x,
                                                epoch_start = self.pars["state"]["n_epoch"])

            # define a predictor to do the evaluations
            #self.predictor = Predictor(self.model, FILTER_ON = True)
            #selr.roller    = Roller(self.predictor, self.policy_obs) 
            # if loop already running initialise from the buffer
            # you might want to sort this out later: if init networks changed, then we need a re-initilisation
            #if reload == True:
            #    pass

        #   Load/Save Admin Part II.

        # load model
        self.saver = tf.train.Saver()
        if os.path.exists(path_model_ckpt + ".meta"):
            self.saver.restore(self.model.sess, path_model_ckpt)
            print("Model %s loaded!" % (path_model_ckpt))

        self.path_exp_yaml           = path_exp_yaml
        self.path_model_ckpt         = path_model_ckpt
        self.counter_current_model   = 0
        self.time_model_update       = time.clock_gettime(time.CLOCK_MONOTONIC)
        self.time_model_update_check = time.clock_gettime(time.CLOCK_MONOTONIC)
        self.counter_burnin          = 0

    def load_model_dummy(self):
        class DummyPredictor():
            def __init__(self, model):
                self.model = model
            def initialize(self, data):
                pass
            def step(self, data):
                return None

        # load model
        self.model = None
        self.predictor  = DummyPredictor(self.model)

        self.saver = None

        self.path_exp_yaml           = None
        self.path_model_ckpt         = None
        self.counter_current_model   = 0
        self.time_model_update       = time.clock_gettime(time.CLOCK_MONOTONIC)
        self.time_model_update_check = time.clock_gettime(time.CLOCK_MONOTONIC)
        self.counter_burnin          = 0

    def run(self,
            CONTROL_ON = False,
            RECORDING_ON = False,
            SYNC_ON=False,
            name_recording = "i_should_not_exist",
            recording_serial = 0):
        """
        Run control loop
        :return:
        """

        INTERACT = True

        self.counter_current_model = 0
        self.counter_life          = 0


        # some stuff for control
        if CONTROL_ON:
            INIT_ON = True
            self.counter_burnin  = 0
            self.wtf_init  = np.zeros((self.model.n_initialobs, 2 * (FrankaConfig.robot.n_joints + FrankaConfig.robot.n_endeff)))

            #  some stuff for masking data / old experiments
            mask_state   = self.model_pars["data"]["mask_state"]
            mask_endeff  = self.model_pars["data"]["mask_endeff"]
            mask_control = self.model_pars["data"]["mask_control"]

        # some control stuff bookkeeping
        control_prev           = np.zeros(self.n_joints)

        counter = 0

        while True:
            # update counters
            self.counter_current_model += 1
            self.counter_life          += 1

            counter += 1

            #if not INTERACT:
            #    time.sleep(0.5)
            #    print((counter,time.clock_gettime(time.CLOCK_MONOTONIC)))

            if True:
                #
                # Request state message
                #
                recv_start = time.clock_gettime(time.CLOCK_MONOTONIC)

                msg_state  = self.broker_server.recv_msg(self.state_channel_id, -1)

                recv_stop  = time.clock_gettime(time.CLOCK_MONOTONIC)

                self.state_time = recv_stop

                # trig_time = time.clock_gettime(time.CLOCK_MONOTONIC)
                # Catch no message for long time
                if (recv_stop - recv_start) > 0.2:
                    print("Reset target")
                    firsttime = 0

                # Extract message A
                q_A           = msg_state.get_j_pos()
                qd_A          = msg_state.get_j_vel()
                endeff_x      = msg_state.get_c_pos()
                endeff_xd     = msg_state.get_c_vel()
                control_prev  = msg_state.get_last_cmd()
                frame_counter = msg_state.get_fnumber()
                recv_time     = msg_state.get_timestamp()

            else:
                q_A             = np.ones(FrankaConfig.robot.n_joints)
                qd_A            = np.ones(FrankaConfig.robot.n_joints)
                endeff_x        = np.ones(FrankaConfig.robot.n_endeff)
                endeff_xd       = np.ones(FrankaConfig.robot.n_endeff)
                control_prev    = np.ones(FrankaConfig.robot.n_joints)
                frame_counter   = counter
                recv_time       = time.clock_gettime(time.CLOCK_MONOTONIC)


            self.endeff_x        = endeff_x
            self.endeff_xd       = endeff_xd
            self.state_q         = q_A
            self.state_qd        = qd_A
            self.state_time      = recv_time
            self.state_recv_time = recv_time
            self.state_fnumber   = frame_counter

            self.control_prev          = control_prev
            self.control_prev_time      = self.state_time
            self.control_prev_recv_time = self.state_recv_time
            self.control_prev_fnumber   = self.state_fnumber


            #
            # Request image message
            #
            if False:
                msg_image          = self.broker_server.recv_msg(self.image_channel_id,-1)
                self.state_img     = msg_image.get_data()
                self.img_time      = time.clock_gettime(time.CLOCK_MONOTONIC)
                self.img_recv_time = msg_image.get_timestamp()
                self.img_fnumber   = msg_image.get_fnumber()

            else:
                self.state_img     = np.zeros(self.state_img_shape)
                self.img_time      = time.clock_gettime(time.CLOCK_MONOTONIC)
                self.img_recv_time = self.img_time
                self.img_fnumber   = counter

            #
            # data processing starts here
            #
            if CONTROL_ON:
                # shape data
                wtf_state = np.concatenate((self.state_q, self.state_qd))
                wtf_endeff = np.concatenate((self.endeff_x, self.endeff_xd))
                wtf_state = np.concatenate((wtf_state[mask_state], wtf_endeff[mask_endeff]))

                if INIT_ON:

                    if True:
                        # if in burnin store data for initialisation
                        if (self.counter_current_model % self.dc_burnin == 0) and (self.counter_burnin < self.model.n_initialobs):
                            self.wtf_init[self.counter_burnin,:]  = wtf_state
                            self.counter_burnin += 1
                            print("Data recorded for init.")

                        # if enough data do initialisation
                        if self.counter_burnin == self.model.n_initialobs:
                            INIT_ON = False

                            if False:
                                print("wtf_init")
                                print(self.wtf_init)

                            if True:
                                # t = time.time()
                                self.predictor.initialize(self.wtf_init)
                                print("Model initilised.")
                                # print(time.time() - t)
                                # print("predictor initialisation successful")
                                #sys.exit()
                else:
                    # update policy
                    if counter % self.dc_policy_update == 0:

                        # if (counter % burnin_step_time == 0):
                        #     print("x_now")
                        #     print(x_now)

                        # do prediction
                        if True:
                            # t = time.time()
                            z_pred_mean, z_pred_var, u_pred_mean, u_pred_var,\
                                x_pred_mean, x_pred_var = self.predictor.step(wtf_state.reshape((1,-1)))

                            mp, vp = self.predictor.get_control()
                            self.policy_mean[mask_control] = mp
                            self.policy_var[mask_control]  = vp

                #
                # Send policy message
                #
                if INTERACT:
                    msg_policy = pab.empowerment_policy_msg()

                    msg_policy.set_fnumber(42)
                    msg_policy.set_j_acc_mean(self.policy_mean)
                    msg_policy.set_j_acc_var(self.policy_var)

                    time_now = time.clock_gettime(time.CLOCK_MONOTONIC)

                    if CONTROL_ON:
                        if (time_now - recv_time) > FrankaConfig.recording.time_lag:
                            time.sleep(time_now - recv_time)

                    msg_policy.set_timestamp(time.clock_gettime(time.CLOCK_MONOTONIC))
                    self.broker_server.send_msg(self.policy_channel_id, msg_policy)

                #
                # Request control message
                #
                if False:
                    msg_control = b.recv_msg(self.control_channel_id, -1)
                    self.control_out  = msg_control.get_torques_des()
                    self.control_time = time.clock_gettime(time.CLOCK_MONOTONIC)

                #
                # check for model update
                #
                if (time.clock_gettime(time.CLOCK_MONOTONIC) - self.time_model_update_check) > FrankaConfig.updates.dt_model_update_check:
                    if os.path.getmtime(self.path_model_ckpt) != self.time_model_update:
                        self.load_model(self.path_exp_yaml, self.path_model_ckpt, reload = True)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", help="[string], directory to load model ckpt from experiment")
    parser.add_argument("-p", help="[string]  model")
    args = parser.parse_args()


    dir_model    = args.d
    dir_policy   = args.p
 
    control_loop = ControlLoop()

    if False::
        model_name      = os.path.split(dir_model)[-1]
        path_model_ckpt = os.path.join(dir_model, model_name)
        path_model_yaml = os.path.join(dir_model, model_name + ".yaml")

        pilcy_name      = os.path.split(dir_policy)[-1]
        path_policy_ckpt = os.path.join(dir_policy, policy_name)
        path_policy_yaml = os.path.join(dir_policy, policy_name + ".yaml")

        control_loop.load_model(path_model_yaml, path_model_ckpt, path_policy_yaml, path_policy_ckpt)
    else:
        control_loop.load_model_dummy()

    control_loop.connect_to_channels()
    control_loop.run(CONTROL_ON = CONTROL_ON,
                     RECORDING_ON = RECORDING_ON,
                     SYNC_ON = SYNC_ON,
                     name_recording = name_recording,
                     recording_serial = recording_serial)

