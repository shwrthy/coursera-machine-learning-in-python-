#Linear Regression
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

#Load the data.
X = loadmat("E:\我的文档\文档\Practices\machine learing - ex\ex1data1.mat")
#You could change the different data to the linear regression training model
X = X['data']
x = np.array(X)
m = x.shape [0]
x0 = np.ones ((m, 1))
x = np.c_[x0, X]
n = x.shape [1]
Y = np.array(x [:, n-1])
y = Y.reshape((m,1)) #To transpose to colomn vector
x = x [:, 0:(n-1)]
theta = np.zeros ((n-1, 1))
alpha = 1e-2 #Learning rate.
times = 1500 #Number of training iteration. You could other number.

#Define the prediction
def prediction_y (theta, x):
    y_pred = np.dot(x, theta)
    return y_pred

#Define the costfunction
def CostFunction (y_pred, y, m):
    cost = (0.5 / m) * (np.power(y_pred - y, 2)).sum()
    return cost

def Optimization (x, y_pred, y, theta, alpha, m):
    delta = np.array(y_pred - y)
    deriv_theta = (1/m) * np.dot(x.T, delta)
    theta = theta - alpha * deriv_theta
    return theta
       
#Define iteration to 
def iteration(x, y, theta, times, m, alpha):
    for iter in range(times):
        y_pred = prediction_y (theta, x)
        theta = Optimization (x, y_pred, y, theta, alpha, m)
    cost = CostFunction (y_pred, y, m)
    print("the Cost Function is: {:.2F} ".format(cost))
    return theta, y_pred

theta, y_pred = iteration(x, y, theta, times, m, alpha)     
plt.scatter(x[:,1:(n-1)],y)
plt.plot(x[:,1:(n-1)], y_pred, label = "$Predition$", color = "green")
plt.show()
