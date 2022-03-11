import numpy as np
from matplotlib import pyplot as plt

m = 11
x = np.array([68,95,102,130,60,45,30,80,120,113,150])
y = np.array([414592,956877,1123582,893667,600000,520000,280000,795000,1150000,1320000,1380234])

#plt.scatter(x,y,color = 'blue',marker='o',label='true')
#plt.show()

w = 8000
b = 100
lr = 0.0001
iltnum = 0

for i in range(100):
    y_pred = w*x+b
    loss = (y_pred -y)**2
    loss = 0.5 * loss.sum()/m
    grad_w = 0.5*np.sum((y_pred -y)*x)/m
    grad_b = 0.5*np.sum(y_pred - y)/m
    w -= lr * grad_w
    b -= lr * grad_b
    iltnum +=1
    print(iltnum,":",loss,grad_w,grad_b,w,b)

plt.plot(x,y_pred,"r-",label="predict value")
plt.scatter(x,y,color = 'blue',marker='o',label='true value')

plt.legend()
plt.show()
print(w,b)
 