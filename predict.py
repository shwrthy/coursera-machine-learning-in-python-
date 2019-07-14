# Instructions: Complete the following code to make predictions using
#               your learned logistic regression parameters. 
#               You should set p to a vector of 0's and 1's
import numpy as np

def predict (theta, X):
    p = np.dot(X, theta)
    m, n = X.shape
    for i in range(m):
        if p[i] > 0:
            p[i] = 1
        else:
            p[i] = 0
    return p