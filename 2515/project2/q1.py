import numpy as np
from sklearn.datasets import load_boston


def Huber_loss(a, delta):
    # a=y-t
    H = np.where(abs(a) <= delta, 1 / 2 * a ** 2, delta * (abs(a) - delta / 2))
    dH = np.where(abs(a) <= delta, a, np.where(a > delta, delta, -delta))
    return H, dH


def gradient_descent(rate, X, y, delta, iteration_time):
    # y target
    # X design matrix
    # w b initial to be zero
    w = np.zeros((1, X.shape[1]))  # w is a row =1 column =size_x
    b = np.zeros((X.shape[0], 1))

    # parameter for huber loss, to get a robust regression

    # loop for gradient descent
    # y is target, set the fx as the current prediction value
    # fx=np.dot(X,w.T)+b

    # while abs(fx-y).all>0.01:
    for i in range(iteration_time):
        fx = np.dot(X, w.T) + b

        a = fx - y  # column =1
        # print(a)
        H, dldw = Huber_loss(a, delta)
        #print(w)
        print("the interation time", i, "  the value is ", np.sum(H))
        # from loss function:
        # dl/dw dl/b
        # we need the cost funciton derivative
        dlw = np.dot(X.T, dldw) / X.shape[0]  # row =1
        w -= (dlw * rate).T
        b -= dldw * rate / X.shape[0]


def main():
    learning_rate = 0.1
    delta = 1
    iteration_time = 20000

    # test 1
    m = 13  # each sample has m size features
    N = 506  # 506 samples
    x = np.random.rand(N, m)
    y = np.zeros((N, 1))
    y.fill(1)
    gradient_descent(learning_rate, x, y, delta, iteration_time)


main()


