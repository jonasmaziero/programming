# in list, order of elements is important

# 2 ways to create a list
#example = list()
#example = []
#print(type(example))

'''
# can create and fill with some values
primes = [2,3,5,7,11,13]
print(primes)
primes.append(17)
primes.append(19)
print(primes)
print('len(primes)=',len(primes))

# access to one or more elements
print(primes[0],type(primes[0]))
print(primes[2:5],type(primes[2:5])) # slicing (the ending value is not included)
# negative indexes
print('primes[-1]=',primes[-1],', primes[-2]=',primes[-2])
'''

# Lists can contain several types of variables, including other lists
#example = [128,True,"Alpha",1.732,[64,False]]
#print(example[0],example[4])

# Lists can contain duplicate values
#rolls = [4,5,2,5,6,4,5]
#print(rolls)


# Combining Lists
numbers = [1,2,3]
letters = ['a','b', 'c']
print(numbers+letters)
print(letters+numbers) # order matters

# more options
print(dir(numbers))
help(numbers.sort)
