import math

example = set()
print(dir(example))
'''
#print(dir(example))
#help(example.add)
example.add(42)
example.add(False)
example.add(math.pi)
example.add("Thorium")
example.add(42) # duplicate items are considered the same
print(example)
#print(type(example))
# the order of the elements in a set does not matter
print(len(example)) # for the number of elements in the set
example.remove(42)
print(example)

# another way to create a set
example2 = set([28,True,math.exp(1),"Helium"])
print(example2)
example2.clear() # shall empty the set
print(len(example2))


# Union and intersection
# example: integers 1 - 10
odds = set([1,3,5,7,9])
print('odds = ',odds)
evens = set([2,4,6,8,10])
print('evens = ',evens)
primes = set([2,3,5,7]) # have only 1 and itself as divisor
print('primes = ',primes)
composites = set([4,6,7,9,10]) # integers that can be factored
print('composites = ',composites)
print('odds U evens = ',odds.union(evens))
print('odds I primes = ',odds.intersection(primes))
print('primes I composites = ',primes.intersection(composites))
print('primes U composites = ',primes.union(composites))
print('4 is 4 prime. ',4 in primes) # test if an element is or not in a set
print('4 is not prime.',4 not in primes)
'''
