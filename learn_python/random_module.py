import random

def random_ab(a,b):
    return (b-a)*random.random() + a

#print(dir(random))
#help(random.random)
#for j in range(10):
    #print(random.random())
    #a = 3; b = 7
    #print(random_ab(a,b))
    #print(random.uniform(a,b)) # uniform prob. distribution
    #mu = 0; sigma = 1
    #print(random.normalvariate(mu,sigma)) # Gaussin prob. dist.
    #i1 = 1; i2 = 6
    #print(random.randint(i1,i2)) # random integers

# picking random samples from a list
outcomes = ['rock','paper','scissors']
for j in range(10):
    print(random.choice(outcomes))
