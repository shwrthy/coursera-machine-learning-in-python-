#Linear Regression
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

## ==================== Part 1: Basic Function ====================
from warmUpExercise import warmUpExercise
print ('Running warmUpExercise ... \n')
print ('5x5 Identity Matrix: \n')
warmUpExercise ()
print ('Program paused. Press enter to continue.\n')
input ()

## ======================= Part 2: Plotting =======================
from plotData import plotData
print ('Plotting Data ...\n')
data = pd.read_csv ('ex1data1.txt', header = None)
data = np.array (data)
X = data [:, 0]
y = data [:, 1]
m = np.size (X, 0) #number of training examples
X = X.reshape ((m, 1))
y = y.reshape ((m, 1))

# Plot Data
plotData(X, y)
print ('Program paused. Press enter to continue.\n')
input ()

## =================== Part 3: Cost and Gradient descent =============
from computeCost import computeCost
X = np.c_ [np.ones((m, 1)), X] # Add a column of ones to x
theta = np.zeros((2, 1)) # initialize fitting parameters

# Some gradient descent settings
iterations = 1500
alpha = 0.01
print ('\nTesting the cost function ...\n')

# compute and display initial cost
J = computeCost(X, y, theta)
print ('With theta = [0 ; 0]\nCost computed = %f\n'%(J))
print ('Expected cost value (approx) 32.07\n')

# further testing of the cost function
J = computeCost(X, y, np.array([-1 , 2]).reshape((2, 1)))
print ('\nWith theta = [-1 ; 2]\nCost computed = %f\n'%(J))
print ('Expected cost value (approx) 54.24\n')
print ('Program paused. Press enter to continue.\n')
input ()

print ('\nRunning Gradient Descent ...\n')
from gradientDescent import gradientDescent

# run gradient descent
theta = gradientDescent (X, y, theta, alpha, iterations)

# print theta to screen
print ('Theta found by gradient descent:\n')
print (theta)
print('Expected theta values (approx)\n')
print ('[[-3.6303],\n [1.1664 ] ]')

from plotData1 import plotData1
#Plot the linear fit
plotData1 (X, y, theta)

# Predict values for population sizes of 35,000 and 70,000
predict1 = np.dot ([1, 3.5], theta) *10000
print ('For population = 35,000, we predict a profit of %f\n'%predict1)
predict2 = np.dot ([1, 7], theta) *10000
print ('For population = 35,000, we predict a profit of %f\n'%predict2)
print ('Program paused. Press enter to continue.\n')
input ()

## ============= Part 4: Visualizing J(theta_0, theta_1) =============
print ('Visualizing J(theta_0, theta_1) ...\n')
# Grid over which we will calculate J
theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 4, 100)

# initialize J_vals to a matrix of 0's
J_vals = np.zeros((len(theta0_vals), len(theta1_vals)))

# Fill out J_vals
for i in range(len(theta0_vals)):
    for j in range(len(theta1_vals)):
        t = np.array([theta0_vals[i], theta1_vals[j]])
        t = t.reshape((2, 1))
        J_vals[i][j] = computeCost(X, y, t)

from mpl_toolkits.mplot3d import Axes3D       
# Because of the way meshgrids work in the surf command, we need to
# transpose J_vals before calling surf, or else the axes will be flipped
J_vals = J_vals.T
theta0_vals, theta1_vals = np.meshgrid(theta0_vals, theta1_vals)
fig = plt.figure(3)
ax = Axes3D(fig)
ax.plot_surface(theta0_vals, theta1_vals, J_vals, rstride=1, cstride=1, cmap='rainbow')
ax.set_xlabel('theta_0')
ax.set_ylabel('theta_1')
ax.set_zlabel('J')
plt.show()

# Contour plot
# Plot J_val-2, 3,s as 15 contours spaced logarithmically between 0.01 and 100
plt.contour(theta0_vals, theta1_vals, J_vals, np.logspace(-2, 3, 20))
plt.plot(theta[0], theta[1], 'rx', linewidth=2, markersize=10)
plt.xlim(-10, 10)
plt.ylim(-1, 4)
plt.xlabel('theta_0')
plt.ylabel('theta_1')
plt.show()
        














