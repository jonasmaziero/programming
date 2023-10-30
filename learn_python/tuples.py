# is a smaller and faster alternative to lists

# list example
#prime_numbers = [2,3,5,7,11,13]
# tuple example
#perfect_squares = (1,4,9,16,25,36)

'''# display lengths
print("# prime numbers = ",len(prime_numbers))
print("# perfect squares = ",len(perfect_squares))

# iterate over the sequence
for p in prime_numbers:
    print("prime number = ",p)
for p in perfect_squares:
    print("perfect square = ",p)'''

# difference between tuples and lists
'''print("list methods") # lists have more functions available
print(dir(prime_numbers))  # lists ocupy more memory than tuples
print("tuple methods")
print(dir(perfect_squares))'''

'''import sys
#print(dir(sys))
print("sizeof prime_numbers = ",sys.getsizeof(prime_numbers),"bytes")
print("sizeof perfect_squares = ",sys.getsizeof(perfect_squares),"bytes")

# better comparison
import math
list_ex = [1,2,3,"a","b","c",True,math.pi]
tuple_ex = (1,2,3,"a","b","c",True,math.pi)
print("sizeof list = ",sys.getsizeof(list_ex),"bytes")
print("sizeof tuple = ",sys.getsizeof(tuple_ex),"bytes")

# also, tuples cannot be changed

import timeit
list_time = timeit.timeit(stmt="[1,2,3,4,5]",number=100000000)
print('list time = ',list_time)
tuple_time = timeit.timeit(stmt="(1,2,3,4,5)",number=100000000)
print('tuple time = ',tuple_time)
print(list_time/tuple_time)'''

# more about tuples
'''empty_tuple = ()
one_element_tuple = ('a',)
not_a_tuple = ('a')
general_tuple = ('a','b','c')
print('empty_tuple = ',empty_tuple)
print('one_element_tuple = ',one_element_tuple)
print('not_a_tuple =',not_a_tuple)
print('general_tuple = ',general_tuple)'''

# alternative construction of tuples
'''tu1 = 1,
tu2 = 1,2
tu3 = 1,2,3
print('tu1 = ', tu1,', tu2 = ', tu2, ', tu3 = ', tu3)'''

# tuple assignement
# (age,country,knows_python)
'''survey = (27,"Vietnam",True)
age = survey[0]
country = survey[1]
knows_python = survey[2]
print("Age = ", age, ", Country = ", country, ", Knows Python = ", knows_python)
survey2 = (21,"Switzerland",False)
age = survey2[0]
country = survey2[1]
knows_python = survey2[2]
print("Age = ", age, ", Country = ", country, ", Knows Python = ", knows_python)'''

# care with the number os variable
x,y,z = (1,2,3) # works
x,y = (1,2,3) # does not work
