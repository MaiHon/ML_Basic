import numpy as np
import matplotlib.pyplot as plt

def numerical_diff(f, x):
    h  = 10e-50
    return(f(x+h)-f(x))/h

# 이 함수는 반올림 오차를 일으킨다.
# 반올림 오차의 예) np.float32(1e-50)
# 너무 작은 값은(소숫점 8자리 이하)가 생략되어 최종결과에도 영향을 미친다.
np.float32(1e-50)

# 개선된 미분
def numerical_diff(f, x):
    h = 10e-4
    return (f(x+h) - f(x-h)) / (2*h)

def function_1(x):
    return 0.01*x**2 + 0.1*x

x = np.arange(0.0, 20.0, 0.1)
y = function_1(x)

plt.xlabel("x")
plt.ylabel("f(x)")
plt.plot(x, y)
numerical_diff(function_1, 5)
numerical_diff(function_1, 10)
# 수치적 미분
# 실제 해석적 미분 값은 2, 3이지만
# 0.29999....6696과 3의 차이는 매우 작은 오차값


# 편미분
def function_2(x):
    #return x[0]**2 + x[1]**2
    return np.sum(x**2)

def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x) # x와 형상이 같은 배열 생성

    for idx in range(x.size):
        tmp_val = x[idx]
        # f(x+h)계산
        x[idx] = tmp_val + h
        fxh1 = f(x)

        # f(x-h)계산
        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val # 값 복원

    return grad

numerical_gradient(function_2, np.array([3.0, 4.0]))
