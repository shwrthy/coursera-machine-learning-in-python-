import numpy as np

# Instructions: Compute the cost of a particular choice of theta.
#               You should set J to the cost.
#               Compute the partial derivatives and set grad to the partial
#               derivatives of the cost w.r.t. each parameter in theta

from sigmoid import sigmoid
                          
def costFunction(initial_theta, X, y):
    m = np.size(y, 0)
    cost = (-np.dot(np.log(sigmoid(np.dot(X, initial_theta))), y)\
        -np.dot(np.log(1 - sigmoid(np.dot(X, initial_theta))),1 - y)) / m
    y = y.reshape((1, m))
    grad = np.dot(sigmoid(np.dot(X, initial_theta))-y, X) / m
    return cost, grad
 
 
     
