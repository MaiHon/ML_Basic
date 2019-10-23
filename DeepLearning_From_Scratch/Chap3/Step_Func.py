import numpy as np
# 계단함수 구현하기

# 활성함수는 입력 신호의 총합을 출력 신호로 변환해주는 함수이다.
# 입력 신호의 총합이 활성화를 일으키는지 정하는 역활을 수행해주는 함수이다.
# 계단함수는 일종의 활성함수(activation function)이다.

# ndarray에도 적용 될 수 있도록 변형
def step_function(x):
    y = x > 0
    return y.astype(np.int8)

x = np.array([-1.0, 1.0, 2.0])
x

step_function(x)


# 계단 함수의 그래프
import matplotlib.pyplot as plt

def step_function(x):
    return np.array(x>0, dtype=np.int)

x = np.arange(-5.0, 5.0, 0.1)
y = step_function(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()
