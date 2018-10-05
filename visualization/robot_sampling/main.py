import visualization.robot_sampling.sampler as sr
import visualization.robot_sampling.controller as cr

import time

import visualization.robot_sampling.lib as lib
import numpy as np
from multiprocessing import Pool, Process, Queue

counter = 0
img_counter = 0
default_pos_c = np.array([0.62, 0.00, 0.56])

n_points = 100
n_runs   = 1
time2go  = 3.
delta    = 2 * np.pi / n_points
R        = 0.2
save_filename = 'data_'

threshold=0.98

data = {}
n_samples=1

save_every = 10
debug = False
total_time = 20 # seconds

runs = -1

broker = lib.default()
def f(broker):
    return broker

counter = lib.pos_msg(default_pos_c, broker, counter)
print("broker initialized")

from multiprocessing import Process, Pool

#process1 = cr.main
#process2 = rt.main


#p1 = Process(target=method1) # create a process object p1
#p1.start()                   # starts the process p1
#p2 = Process(target=method2)
#p2.start()

#l1 = Queue()
#print("l1")
#p1 = Process(target=method1, args=(broker, l1, ))  
#l2 = Queue()
#print("l2")
#p2 = Process(target=method2, args=(broker, l2, )) 
#p1.start()   

#result = map(f, [cr.main, rt.main])
#print(result)
#print("ok")
#print(result)

# pool = Pool()

# p1 = pool.apply_async(process1, [broker])
# p2 = pool.apply_async(process2, [broker])

# #pool.close()
# res = [p1, p2]

# final = [p.get() for p in res]
# print(final)


def p1_f(b):
    res = sr.main(b)
    return res

def p2_f(b):
    res = cr.main(b)
    return res


from concurrent.futures import ThreadPoolExecutor

print("Starting ThreadPoolExecutor")
with ThreadPoolExecutor(max_workers=4) as executor:
    future = executor.submit(p1_f, (broker))
    future = executor.submit(p2_f, (broker))

print(future)
print("All tasks complete")
