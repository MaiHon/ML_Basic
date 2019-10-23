import numpy as np
import matplotlib.pyplot as plt
# 평균제곱오차 Mean Squared Error
# 가장 많이 쓰이는 손실함수
# 신경망 학습은 하나의 지표를 기준으로 최적의 매개변수 값을 탐색한다.
# 손실함수를 그 지표로서 사용한다.

def mean_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)

t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
mean_squared_error(np.array(y), np.array(t))

# 교차 엔트로피 오차 Cross Entropy error(CEE)

def cross_entropy_error(y, t):
    # y값이 0이 되면 inf값을 리턴하기 때문에 아주 작은 delta값 더해준다.
    delta = 1e-7
    return -np.sum(t*np.log(y+delta))

t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
y1 = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
y2 = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]

cross_entropy_error(np.array(y1), np.array(t))
cross_entropy_error(np.array(y2), np.array(t))
# t값이 1인 부분(정답레이블)의 확률이 더낮은 0.1로 바뀌면
# 오차값이 0.51에서 2.3으로 커지는것을 확인 할 수 있다.


# random으로 m미만의 수 중에서 n개의 데이터 뽑아내기
# np.random.choice(m, n)
np.random.choice(60000, 10)


# 배치학습을 위한 cross_entropy_error함수 수정
# 원핫코딩
def cross_entropy_error(y, t):
    # 데이터 하나당 교차엔트로피손실을 구하는 경우
    # 즉 y의 차원이 1차원일 경우
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # 그 외의 경우 1개 이상의 배치사이즈를 가지면
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y+1e-7)) / batch_size

# 원핫코딩이 아닌 출력값 그대로 넣을경우
def cross_entropy_error(y, t):
    # 데이터 하나당 교차엔트로피손실을 구하는 경우
    # 즉 y의 차원이 1차원일 경우
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # 그 외의 경우 1개 이상의 배치사이즈를 가지면
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t]+1e-7))/batch_size

# 신경망 학습에서 정확도를 지표로 삼으면 안됨
# 정확도를 지표로 삼을시에는 매개변수의 미분이 대부분 장소에서 0이 되기 때문에
# 매개변수의 손실 함수의 미분이란 '가중치 매개변수의 값을 아주 조금 변화 시켰을 때 손실함수가 어떻게 변하는가'를 의미한다.
# 이 미분값이 음수라면 가중치 매개변수의 값을 양으로 변화시켜 손실함수의 값을 줄일 수 있다.
