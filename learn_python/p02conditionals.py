import numpy as np # para usar conjuntos de funções prontas
import math

def bascara():
    print("Forneça os coeficientes de ax^2+bx+c=0")
    a = float(input('Digite a: '))
    b = float(input('Digite b: '))
    c = float(input('Digite c: '))
    Delta = pow(b,2)-4*a*c
    print("Delta =", Delta)
    print("Raízes")
    if Delta == 0:
        x = -b/(2*a)
        print("x1 = x2 = ", x.real)
    elif Delta > 0:
        x1 = (-b + math.sqrt(Delta))/(2*a)
        x2 = (-b - math.sqrt(Delta))/(2*a)
        print("x1 = ", x1.real, "  x2 = ", x2.real)
    elif Delta < 0:
        x = -b/(2*a) + 1j*math.sqrt(abs(Delta))/(2*a)
        print("x1 =", x.real, "+I*", x.imag, ", x2 =", x.real, "-I*", x.imag)


#bascara()
