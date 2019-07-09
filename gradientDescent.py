import numpy as np
import pandas as pd
# Instructions: Perform a single gradient step on the parameter vector
#               theta.

def gradientDescent (X, y, theta, alpha, iterations):
    m = np.size (X, 0)
    for iter in range(iterations):
        delta = np.dot(X, theta) - y
        theta = theta - alpha / m * np.dot(X.T, delta)
        
    return theta
    
