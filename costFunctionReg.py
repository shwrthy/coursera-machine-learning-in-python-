import numpy as np

# Instructions: Compute the cost of a particular choice of theta.
#               You should set J to the cost.
#               Compute the partial derivatives and set grad to the partial
#               derivatives of the cost w.r.t. each parameter in theta
from sigmoid import sigmoid

def costFunctionReg(theta, X, y, Lambda):
    m, n = X.shape
    J = (-np.dot(np.log(sigmoid(-np.dot(X, theta))), y) -np.dot(np.log(1 - \
         sigmoid(-np.dot(X, theta))),1 - y)) / m + Lambda / m / 2 * \
         np.power(np.multiply(theta, np.c_[0, np.ones((1, n-1))]), 2).sum()

    y = y.reshape((1, m))
    grad = np.dot(sigmoid(-np.dot(X, theta))-y, X) / m + \
    Lambda / m * np.multiply(theta, np.c_[0, np.ones((1, n-1))])
    return J, grad
