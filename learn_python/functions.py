def ping():
    return "Ping"

#x = ping(); print(x)

import math
def sphere_volume(r):
    return (4/3)*math.pi*r**3

#print("pi = ",math.pi)
#print(sphere_volume(1))


def cm(feet = 0, inches = 0): # default arguments are called keyword arguments
    """Converts a length from   feet and inches to centimeters"""
    # 1 inch = 2.54 cm
    # 1 foot = 12 inches
    inches_to_cm = inches*2.54
    feet_to_cm = feet*12*2.54
    return inches_to_cm + feet_to_cm

#print(cm(feet = 5))
#print(cm(inches = 8))
#print(cm(feet = 5, inches = 8))

# keyword and required arguments
def fargs(x, y = 0): # x must go first (it is a required argument)
    return x+y

print(fargs(1,y=2))
print(fargs(x=1,y=2))
print(fargs(1,2))
#print(fargs()) # does not work
