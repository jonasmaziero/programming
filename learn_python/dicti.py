'''
import numpy as np
f = open("/Users/jonasmaziero/Dropbox/Research/ibm/bds/calc/dados_plot/dados_plot1/0/XY.dat", "r")
data = {}
data = f.read()
f.close()
print(data)
print(data[8:12])
print(data[21:25])
print(data[35:39])
print(data[49:53])
pXX = np.zeros((4,2),dtype=int)
pXX[0][1] = int(data[8:12])
pXX[1][1] = int(data[21:25])
pXX[2][1] = int(data[35:39])
pXX[3][1] = int(data[49:53])
print(pXX)
'''
arquivo = open(fname, 'r')
for line in arquivo:
  cols =  line.split(' ')
a = cols[1]; b = cols[3]; c = cols[5]; d = cols[7]
a = a[:-1]; b = b[:-1]; c = c[:-1]; d = d[:-1]
pXX = np.zeros((4,2))
pXX[0][1] = int(a)
pXX[1][1] = int(b)
pXX[2][1] = int(c)
pXX[3][1] = int(d)
