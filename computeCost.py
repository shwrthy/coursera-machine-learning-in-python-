import numpy as np
# Instructions: Compute the cost of a particular choice of theta
#               You should set J to the cost.
def computeCost(X, y, theta):
    m = np.size (X, 0)
    delta = np.dot(X, theta) - y
    delta = np.power(delta, 2)
    return 0.5 / m * delta.sum()
