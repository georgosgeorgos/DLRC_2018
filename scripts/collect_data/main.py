import time
import numpy as np
from multiprocessing import Pool, Process, Queue
from concurrent.futures import ThreadPoolExecutor
#__________________________________________________
import lib
import sampler
import controller
#__________________________________________________
broker = lib.default()
print("broker initialized")


def sampler_f(broker):
    res = sampler.main(broker)
    return res

def controller_f(broker):
    res = controller.main(broker)
    return res

print("Starting ThreadPoolExecutor")
with ThreadPoolExecutor(max_workers=4) as exe:
    future = exe.submit(sampler_f, (broker))
    future = exe.submit(controller_f, (broker))

print(future)
print("All tasks completed")
#__________________________________________________


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
