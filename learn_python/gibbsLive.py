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
ax = plt.axes(xlim=(0, 20), ylim=(0, 1))
line, = ax.plot([], [], lw=2)


def init():
    line.set_data([], [])
    return line,


def animate(i):
    x = np.linspace(0, 20, 20)
    y = (1-np.exp(-0.01*i))*(np.exp(-0.01*i))**x
    line.set_data(x, y)
    return line,


ani = animation.FuncAnimation(fig, animate, init_func=init, frames=300,
                              interval=50, blit=True)

#ani.save('ani.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

plt.show()
