from numpy import *


def sigmoid(X, deriv=False):
    if deriv == True:
        return X * (1 - X)
    return 1/(1 + exp(-X))


def cost(X, y, theta):
    a = X * theta
    y_out = sigmoid(a)
    m = len(X)
    J = -(multiply(y, log(y_out)) + multiply((1 - y), log(1 - y_out))) / m
    return sum(sum(J))


def grad(X, y, theta):
    a = X * theta
    y_out = sigmoid(a)
    m = len(X)
    grad = zeros_like(theta)
    print(sum((y_out - y))
#     for i in range(len(theta)):
#         grad[i] = [1 / m * multiply(sum((y_out - y), X[:, i]))]
#     return grad

def main():
    data=genfromtxt("data1.csv", delimiter=",")

    X=data[:, [0, 1]]
    y=data[:, [2]]

    m=len(X)
    X=append(ones((m, 1)), X, axis=1)

    theta=array([0, 0, 0]).T
    print(cost(X, y, theta))
    grad(X, y, theta)


if __name__ == '__main__':
    main()
