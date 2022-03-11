import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-1,6,141)
y = (x-2.5)**2-1

def dF(theta):
    return 2*(theta-2.5)

def F(theta):
    return (theta-2.5)**2-1

theta = 0.0
learn_rate = 0.1
epsilon = 1e-8
p_history = []
litnum = 0
while True:
    gradient = dF(theta) 
    last_theta = theta
    theta = theta - learn_rate*gradient
    p_history.append(theta)

    litnum += 1
    if(abs(F(last_theta)-F(theta))<epsilon):
        break

plt.plot(x,y)
plt.plot(np.array(p_history), F(np.array(p_history)), color = 'r', marker='+')
plt.show()

print(litnum,":",theta,F(theta))
