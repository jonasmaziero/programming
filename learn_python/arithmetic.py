'''
a = 13
print(type(a),a)
b = float(a)
print(type(b),b)
c = 3.1415
print(type(c),c)
d = int(c)
print(type(d),d)
e = complex(c)
print(type(e),e)
#float(e) # does not work
'''

# complex is wider than float that is wider than int
# In + - + / operations, transformation to wider classes are done
a = 1
b = 2.0
c = 3 + 0j
print(b/c,2/3) # transforms to complex and divide a/b = a*cj(b)/b*cj(b)
print('20/5',20/5) # na divisão transforma-se as variáveis para real ou complexo

print('16/5',16/5) # em Python3 não tem aredondamento
print('16//5',16//5) # para ter aredondamento
print('19/5',19/5)
print('19//5',19//5) # aredondamento é feito pra 'baixo'

print(16%5) # resto da divisão

#print(1/0) # cuidado
