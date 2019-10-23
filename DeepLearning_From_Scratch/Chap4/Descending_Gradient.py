import numpy as np
# 최적: 손실함수가 최소값이 될 때의 매겨변수 값
# 기울기를 이용해서 함수의 최솟값을 찾으려는 것이 경사법

# 함수가 극솟값, 최솟값 또는 안정점이 되는 장소에서는 기울기가 0
# 기울어진 방향이 꼭 최솟값을 가리키는 것은 아니지만, 그 방향으로
# 가야만 값을 줄일 수 있다.

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

# lr(learning ratio) 학습률
# 한 번의 학습으로 얼마만큼 학습해야 할지를 나타내는 지표
# 매개변수 값을 얼마나 갱신할지를 정하는 것이 학습률
def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x.copy()

    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad

    return x

def function_2(x):
    return np.sum(x**2)

init_x = np.array([-3.0, 4.0])
print(gradient_descent(function_2, init_x, lr=10.0, step_num=100))
print(gradient_descent(function_2, init_x))
print(gradient_descent(function_2, init_x, lr=0.1))
# 학습률이 너무 크면 오차값이 발산해버린다.
# 반대로 너무 작으면 학습이 제대로 이루어 지지 않고 종료되버린다.
# 학습률을 적절히 조절하는 것이 매우 중요하다.
