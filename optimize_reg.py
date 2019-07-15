import pandas as pd
import numpy as np
from scipy.optimize import minimize

def sigmoid (x):
    z = 1 / (1 + np.exp(-x))
    return z

def costFunction(initial_theta, X, y):
    m = np.size(y, 0)
    J = (-np.dot(np.log(sigmoid(np.dot(X, initial_theta))), y)\
        -np.dot(np.log(1 - sigmoid(np.dot(X, initial_theta))),1 - y)) / m
    return J
 
 
def gradient(initial_theta, X, y):
    m, n = X.shape
    y = y.reshape((1, m))
    grad = np.dot(sigmoid(np.dot(X, initial_theta))-y, X) / m
    return grad

def costFunction_Reg (theta, X, y, Lambda):
    m, n = X.shape
    J = (-np.dot(np.log(sigmoid(np.dot(X, theta))), y) -np.dot(np.log(1 - \
         sigmoid(np.dot(X, theta))),1 - y)) / m + Lambda / m / 2 * \
         np.power(np.multiply(theta, np.insert(np.ones((n-1)), 0, 0.)), 2).sum()
    return J

def gradient_Reg (theta, X, y, Lambda):
    m, n = X.shape
    y = y.reshape((1, m))
    grad = np.dot(sigmoid(np.dot(X, theta))-y, X) / m + \
    Lambda / m * np.multiply(theta, np.insert(np.ones((n-1)), 0, 0.))
    return grad


data = pd.read_csv ('ex2data2.txt', header = None)
data = np.array (data)
X = data [:, 0:2]
y = data [:, 2]
from mapFeature import mapFeature
X = mapFeature (X[:,0], X[:,1])
m, n = X.shape
y = y.reshape((m, 1))
# Initialize fitting parameters
initial_theta = np.zeros(n)
# Set regularization parameter lambda to 1
Lambda = 1
cost = costFunction_Reg (initial_theta, X, y, Lambda)
grad = gradient_Reg (initial_theta, X, y, Lambda)
result = minimize (fun = costFunction_Reg, x0 = initial_theta, \
                   args = (X, y, Lambda), method = 'TNC', jac = gradient_Reg)