from numpy import *


def sigmoid(X, deriv=False):
    if deriv == True:
        return X * (1 - X)
    return 1/(1 + exp(-X))


def cost(X, y, theta):
    a = dot(X, theta)
    y_out = sigmoid(a)
    m = len(X)
    J = -(multiply(y, log(y_out)) + multiply((1 - y), log(1 - y_out))) / m
    return sum(sum(J))

def main():
    data = genfromtxt("data1.csv", delimiter=",")

    X = data[:, [0, 1]]
    y = data[:, [2]]

    m = len(X)
    n = size(X, axis=1)
    X = append(ones((m, 1)), X, axis=1)

    start_theta = zeros((n+1, 1))
    print(cost(X, y, start_theta))
    # grad(X, y, theta)


if __name__ == '__main__':
    main()
