import numpy as np
import matplotlib.pyplot as plt

# 계단함수는 0과 1의 극단적인 활성값만 던져준다.
# 반면 시그모이드함수는 0 ~ 1까지 각 상황에 맞는 확률값을 던져준다.

"""
    Sigmoid activation function
    h(x) = 1 / (1+exp(-x))
"""

def sigmoid(x):
    return 1/(1+np.exp(-x))


t = np.array([-1.0, 1.0, 2.0])
sigmoid(t)


x = np.arange(-5.0, 5.0, 1.0)
y = sigmoid(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1) # 
plt.show()
