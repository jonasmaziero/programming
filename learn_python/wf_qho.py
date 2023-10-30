import math

def exp(x):
    return math.exp(x)

def sqrt(x):
    return math.sqrt(x)

def wf(alpha,u,n):
    pi = math.pi
    return sqrt(2)*alpha**(1/4)*u*exp(-u**2/2)/pi**(1/4)

print(wf(3,1,1))