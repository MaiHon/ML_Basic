import numpy as np
# Identity Function: 항등함수
# 입력과 출력이 항상 같다


# Softmax Function: 소프트맥스함수
# 분류에서 사용한다.
# normalizes it into a probability distribution

# 소프트맥스 함수의 출력은 "확률"이다.
# 소프트맥스 함수의 출력은 항상 0~1 사이의 값
# 출력의 총합은 항상 1
# 소프트맥스 함수를 이용함으로써 문제를 확률적, 통계적으로 대응할 수 있음

def softmax(a):
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)

    return exp_a/sum_exp_a

# 위 함수는 컴퓨터로 계산할 때는 결함이 있다.
# 오버플로우 문제
# np.exp(1000) = inf로 결과값이 돌아옴
# 이렇게 큰 값끼리 나눗셈을 하면 결과 수치가 "불안정"해짐

def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c) # 오버플로우 대책
    sum_exp_a = np.sum(exp_a)

    return exp_a/sum_exp_a


a = np.array([1010, 1000, 900])
np.exp(a)

c = np.max(a)
np.exp(a-c)


a = np.array([0.3, 2.9, 4.0])
y = softmax(a)
print(y)
np.sum(y)
