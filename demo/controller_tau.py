from time import sleep, time
import scripts.collect_data.lib as lib
import numpy as np
import sys
import time
import py_at_broker as pab
sys.path.append('/home/georgos/DLRC_2018/')

ROBOT_NAME = 'franka'
broker = pab.broker()
time.sleep(0.2)
print("Register signal <{}_des_tau> {}".format(ROBOT_NAME, broker.register_signal(ROBOT_NAME + "_des_tau",
                                                                                  pab.MsgType.des_tau)))
time.sleep(0.2)

counter = 0

print("Press ENTER")
b=input()
while True:
    try:
        counter=lib.tau_msg(broker, counter)
        print("\rposition counter: {}".format(counter), end="")
        sys.stdout.flush()
    except KeyboardInterrupt:
        break