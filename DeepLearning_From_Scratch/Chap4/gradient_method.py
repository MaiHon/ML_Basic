# coding: utf-8
import numpy as np
import matplotlib.pylab as plt
from Chap4.gradient_2d import numerical_gradient


def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x.copy()
    x_history = []

    for i in range(step_num):
        x_history.append( x.copy() )

        grad = numerical_gradient(f, x)
        x -= lr * grad

    return x, np.array(x_history)


def function_2(x):
    return x[0]**2 + x[1]**2

init_x = np.array([-3.0, 4.0])

step_num = 20
x1, x_history1 = gradient_descent(function_2, init_x, lr=0.1, step_num=step_num)
x2, x_history2 = gradient_descent(function_2, init_x, lr=0.01, step_num=step_num)
x3, x_history3 = gradient_descent(function_2, init_x, lr=10.0, step_num=step_num)

plt.plot([-5, 5], [0,0], '--b')
plt.plot([0,0], [-5, 5], '--b')
plt.plot(x_history1[:,0], x_history1[:,1], 'bo')
plt.plot(x_history2[:,0], x_history2[:,1], 'go')
plt.plot(x_history3[:,0], x_history3[:,1], 'ro')

plt.xlim(-3.5, 3.5)
plt.ylim(-4.5, 4.5)
plt.show()
