#%matplotlib inline
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
'''N = 10
x = np.zeros(N)
y = np.zeros(N)
bho = 10
for j in range(0,N):
    x[j] = j
    y[j] = (1-np.exp(-bho))*(np.exp(-bho))**j
plt.bar(x, y, label = '', color = 'blue')
plt.xlabel('j')
plt.ylabel('pj')
plt.legend()
axes = plt.gca()
axes.set_xlim([0-0.5,N])
axes.set_ylim([0,1])
'''
fig = plt.figure()
ax = plt.axes(xlim=(0, 2), ylim=(-2, 2))
line, = ax.plot([], [], lw=2)


def init():
    line.set_data([], [])
    return line,


def animate(i):
    x = np.linspace(0, 2, 1000)
    y = np.sin(2*np.pi*(x-0.01*i))
    line.set_data(x, y)
    return line,


ani = animation.FuncAnimation(fig, animate, init_func=init, frames=200,
                              interval=20, blit=True)

#ani.save('ani.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

plt.show()
