# collect string, test length

'''
inp = input("Please, enter a test string: ")
if len(inp) < 6:
    print('Your string is too short.')
'''
'''
inp = input("Enter an integer:") # input is of type string
number = int(inp)
if number % 2 == 0:
    print('even')
else:
    print('odd')
'''

a = int(input("Enter the length of side a: "))
b = int(input("Enter the length of side b: "))
c = int(input("Enter the length of side c: "))

if a != b and b != c and a != c:
    print("scalene triangle")
elif a == b and b == c:
    print("equilateral triangle")
else:
    print("isosceles triangle")
