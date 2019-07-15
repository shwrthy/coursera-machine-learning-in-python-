import numpy as np
def mapFeature (X1, X2):
# MAPFEATURE Feature mapping function to polynomial features
#   MAPFEATURE(X1, X2) maps the two input features
#   to quadratic features used in the regularization exercise.
#   Returns a new feature array with more features, comprising of 
#   X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..
#   Inputs X1, X2 must be the same size
    
    m = X1.shape[0]
    X1 = X1.reshape((m, 1))
    X2 = X2.reshape((m, 1))
    out = np.array(np.ones((m,1)))
    degree = 6
    for i in range (1,degree + 1):
        for j  in range (i + 1):
            out1 = np.multiply(np.power(X1, i-j), np.power(X2, j))
            out = np.c_[out, out1]
    return out
