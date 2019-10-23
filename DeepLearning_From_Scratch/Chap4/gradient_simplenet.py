import sys, os
sys.path.append(os.pardir)
import numpy as np
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient

class simplenet:
    def __init__(self):
        self.W = np.random.randn(2, 3)

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)

        return loss


def f(W):
    return net.loss(x, t)

if __name__ == "__main__":
    net = simplenet()
    print(net.W)

    x = np.array([0.6, 0.9])
    p = net.predict(x)
    print(p)
    np.argmax(p) # 최댓값의 인덱스 얻기
    t = np.array([0, 0, 1]) # 정답 레이블
    net.loss(x, t)

    f = lambda w: net.loss(x, t)
    dW = numerical_gradient(f, net.W)
    print(dW)
