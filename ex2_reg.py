# Logistic Regression in Regularization
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize

## Load Data
#  The first two columns contains the X values and the third column
#  contains the label (y).
data = pd.read_csv ('ex2data2.txt', header = None)
data = np.array (data)
X = data [:, 0:2]
y = data [:, 2]
from plotData import plotData
plotData(X, y)
plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip Test 2')
plt.legend (['y = 1', 'y = 0'])
plt.show ()

## =========== Part 1: Regularized Logistic Regression ============
#  In this part, you are given a dataset with data points that are not
#  linearly separable. However, you would still like to use logistic
#  regression to classify the data points.
#  To do so, you introduce more features to use -- in particular, you add
#  polynomial features to our data matrix (similar to polynomial
#  regression).

# Note that mapFeature also adds a column of ones for us, so the intercept
# term is handled
from mapFeature import mapFeature
X = mapFeature (X[:,0], X[:,1])
m, n = X.shape
y = y.reshape((m, 1))
# Initialize fitting parameters
initial_theta = np.zeros(n)
# Set regularization parameter lambda to 1
Lambda = 1
# Compute and display initial cost and gradient for regularized logistic
# regression
from costFunctionReg import costFunctionReg
cost, grad = costFunctionReg (initial_theta, X, y, Lambda)
print ('Cost at initial theta (zeros): %f\n'% cost)
print ('Expected cost (approx): 0.693\n')
print ('Gradient at initial theta (zeros) - first five values only:\n')
print (grad[:,0:5])
print ('Expected gradients (approx) - first five values only:\n')
print (' 0.0085\n 0.0188\n 0.0001\n 0.0503\n 0.0115\n')
print ('\nProgram paused. Press enter to continue.\n')
input ()

# Compute and display cost and gradient
# with all-ones theta and lambda = 10
test_theta = np.ones(n)
cost1, grad1 = costFunctionReg(test_theta, X, y, 10)
print ('\nCost at test theta (with lambda = 10): %f\n' % cost1)
print ('Expected cost (approx): 3.16\n')
print ('Gradient at test theta - first five values only:\n')
print (grad1[:,0:5])
print ('Expected gradients (approx) - first five values only:\n')
print (' 0.3460\n 0.1614\n 0.1948\n 0.2269\n 0.0922\n')
print ('\nProgram paused. Press enter to continue.\n')
input ()

## ============= Part 2: Regularization and Accuracies =============
#  Optional Exercise:
#  In this part, you will get to try different values of lambda and
# see how regularization affects the decision coundart
#  Try the following values of lambda (0, 1, 10, 100).
#  How does the decision boundary change when you vary lambda? How does
#  the training set accuracy vary?

initial_theta = np.zeros(n) # Initialize fitting parameters
Lambda = 1 # Set regularization parameter lambda to 1 (you should vary this)
# Optimize
from multiFunction import optimize_Reg
result = optimize_Reg (initial_theta, X, y, Lambda)
theta = result.x
from plotDecisionBoundary import plotDecisionBoundary
plotDecisionBoundary(theta, X, y)
plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip Test 2')
plt.legend (labels = ['y = 1','y = 0','bonundary'])
plt.show ()
# Compute accuracy on our training set
from predict import predict
p = predict(theta, X)
p = p.reshape ((m, 1))
ans = np.mean(np.double(p == y)) * 100
print ('Train Accuracy: %f\n'% ans)
print ('Expected accuracy (with lambda = 1): 83.1 (approx)\n')
