# Fibonacci sequence: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, ...

# Without memoization
def fibonacci__(n):
    """Returns the n-th term in Fibonacci's sequence"""
    if n == 0:
        return 0
    elif n == 1:
        return 1
    elif n == 2:
        return 1
    elif n > 2:
        return fibonacci(n-1)+fibonacci(n-2) # recursion (it is calling itself)
    # this code is slow, because multiple evaluations are made for a given n

# Memoization with a cache
fibonacci_cache = {}
def fibonacci_(n):
    """Returns the n-th term in Fibonacci's sequence"""
    """Now using memoization"""
    if n in fibonacci_cache:
        return fibonacci_cache[n]
    if n == 0:
        value = 0
    elif n == 1:
        value = 1
    elif n == 2:
        value = 1
    elif n > 2:
        value = fibonacci(n-1)+fibonacci(n-2)
    fibonacci_cache[n] = value
    return value

# Python builtin memoization
from functools import lru_cache
@lru_cache(maxsize = 1000)
def fibonacci(n):
    """Returns the n-th term in Fibonacci's sequence"""
    if type(n) != int:
        raise TypeError("n must be a nonnegative integer")
    if n < 0:
        raise ValueError("n must be a nonnegative integer")
    if n == 0:
        return 0
    elif n == 1:
        return 1
    elif n == 2:
        return 1
    elif n > 2:
        return fibonacci(n-1)+fibonacci(n-2)

'''
import time
import timeit
#ti = time.time()
#ti = timeit.default_timer()
ti = time.process_time()
for n in range(0,100000):
    #print(n,":",fibonacci__(n))
    fib = fibonacci__(n)
#tf = time.time()
#tf = timeit.default_timer()
tf = time.process_time()
print('time without memoization = ', tf-ti,'seconds') # 0.59 s

#ti = time.time()
#ti = timeit.default_timer()
ti = time.process_time()
for n in range(0,100000):
    #print(n,":",fibonacci_(n))
    fib = fibonacci_(n)
#tf = time.time()
#tf = timeit.default_timer()
tf = time.process_time()
print('time with cache memoization = ', tf-ti,'seconds') # 1.76 s

#ti = time.time()
#ti = timeit.default_timer()
ti = time.process_time()
for n in range(0,100000):
    #print(n,":",fibonacci(n))
    fib = fibonacci(n)
#tf = time.time()
#tf = timeit.default_timer()
tf = time.process_time()
print('time with auto memoization = ', tf-ti,'seconds') # 0.46 s
'''

# handling type or value errors
#print(fibonacci(2))


for n in range(1,51):
    #print(fibonacci(n)) # the value raises quickly
    print(fibonacci(n+1)/fibonacci(n)) # = golden ratio = 1.618033988749895
