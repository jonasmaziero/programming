import multiprocessing as mp 

#print('No. of CPU cores = ', mp.cpu_count())

'''
# The process class
# workflow: create object -> start the process -> join the process (to terminate)
def my_func():
    print("Hello multiWorld")

proc = mp.Process(target=my_func)
proc.start() # to start the process
proc.join() # to terminate the process
'''

'''
# creaing multiple processes
def lang_func(lang):
    print(lang)

langs = ['C', 'Python', 'Fortran', 'Latex']
processes = [] # empty list
for l in langs:
    proc = mp.Process(target=lang_func, args=(l,))
    processes.append(proc)
    proc.start()
for p in processes:
    proc.join()
'''

# The Lock class: 

# The Queue class: interprocess communication
def sqr(x,q):
    q.put(x*x) # to put the data to the queue (we add to add the q arg)
q = mp.Queue()
'''
p = mp.Process(target=sqr, args=(4,q))
p.start()
p.join()
result = q.get() # get the data from the queue 
print(result)

# now with multiple processes
processes = [mp.Process(target=sqr, args=(i,q)) for i in range(2,10)]
for p in processes:
    p.start()
for p in processes:
    p.join()
result = [q.get() for p in processes]
print(result)
'''

# the Pool class: ???
import time
tasks = (["A",5],["B",2],["C",1],["S",3])
def tasks_exec(tasks_data):
    print(f'Process {tasks_data[0]} waiting {tasks_data[1]} seconds')
    time.sleep(int(tasks_data[1]))
    print(f'Process {tasks_data[1]} finished')
def pool_func():
    p = mp.Pool(4)
    p.map(tasks_exec, tasks)
pool_func()