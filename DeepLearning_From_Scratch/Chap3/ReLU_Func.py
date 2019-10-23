import numpy as np
import matplotlib.pyplot as plt

# ReLu(Rectified Linear Unit)함수
# 입력이 0을 넘으면 그 입력을 그대로 출력하고,
# 입력이 0이하이면 0을 출력해준다.

# Sigmoid활성함수는 은닉층(Hidden layer)가
# 깊어질 수록 출력값이 급속도록 낮아져서 올바른 결과가
# 나오지 않을 수가 있다. -> ReLU함수로 극복

def relu(x):
    return np.maximum(0, x)


x = np.arange(-5.0, 5.0, 1.0)
y = relu(x)
plt.plot(x, y)
plt.ylim(-0.1, 4.0)
plt.xlim(-5.0, 4.0)
plt.ylabel("Output")
plt.xlabel("Input")
plt.show()
