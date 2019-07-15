
# Logistic Regression
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize

## Load Data
#  The first two columns contains the exam scores and the third column
#  contains the label.
data = pd.read_csv ('ex2data1.txt', header = None)
data = np.array (data)
X = data [:, 0:2]
y = data [:, 2]

## ==================== Part 1: Plotting ====================
#  We start the exercise by first plotting the data to understand the 
#  the problem we are working with.
from plotData import plotData
print (['Plotting data with + indicating (y = 1) examples \
        and o indicating (y = 0) examples.\n'])
plotData(X, y)
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.legend (['Admitted', 'Not admitted']) # Automatic detectson of elements to be shown in the legend
plt.show ()
print ('\nProgram paused. Press enter to continue.\n')
input ()


## ============ Part 2: Compute Cost and Gradient ============
#  In this part of the exercise, you will implement the cost and gradient
#  for logistic regression. You neeed to complete the code in 
#  costFunction.py
# Setup the data matrix appropriately, and add ones for the intercept term
m, n = X.shape
y = y.reshape((m, 1))
X = np.c_ [np.ones ((m, 1)), X] # Add intercept term to x and X_test
initial_theta = np.zeros ((n + 1)) # Initialize fitting parameters
# Compute and display initial cost and gradient
from multiFunction import costFunction
from multiFunction import gradient
cost = costFunction (initial_theta, X, y)
grad = gradient (initial_theta, X, y)
print ('Cost at initial theta (zeros): %f\n' % cost)
print ('Expected cost (approx): 0.693\n')
print ('Gradient at initial theta (zeros): \n')
print (grad)
print ('Expected gradients (approx):\n[ [-0.1000]\n  \
       [-12.0092]\n  [-11.2628]]\n')
# Compute and display cost and gradient with non-zero theta
test_theta = np.array([-24, 0.2, 0.2])
cost = costFunction(test_theta, X, y)
grad = gradient (test_theta, X, y)
print ('\nCost at test theta: %f\n' % cost)
print ('Expected cost (approx): 0.218\n')
print ('Gradient at test theta: \n')
print (grad)
print ('Expected gradients (approx):\n[ [0.043]\n  [2.566]\n  [2.647]]\n')
print ('\nProgram paused. Press enter to continue.\n')
input ()

##  ============= Part 3: Optimizing using fminunc  =============
#  In this exercise, you will use a built-in function (fminunc) to find the
#  optimal parameters theta.
# Set options for fminunc
from multiFunction import optimize
result = optimize (initial_theta, X, y)
print ('Cost at theta found by fminunc: %f\n' % result.fun)
print ('Expected cost (approx): 0.203\n')
print ('theta: \n')
theta = result.x
print (theta)
print ('Expected theta (approx):\n')
print ('[ [-25.161]\n  [0.206]\n  [0.201]\n')

# Plot Boundary
from plotDecisionBoundary import plotDecisionBoundary
plotDecisionBoundary(theta, X, y)
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.legend (['Bounday','Admitted', 'Not admitted']) # Automatic detectson of elements to be shown in the legend
plt.show ()

## ============== Part 4: Predict and Accuracies ==============
#  After learning the parameters, you'll like to use it to predict the outcomes
#  on unseen data. In this part, you will use the logistic regression model
#  to predict the probability that a student with score 45 on exam 1 and 
#  score 85 on exam 2 will be admitted.
#  Furthermore, you will compute the training and test set accuracies of 
#  our model.
#  Your task is to complete the code in predict.m
#  Predict probability for a student with score 45 on exam 1 
#  and score 85 on exam 2 
from multiFunction import sigmoid
prob = sigmoid(np.dot(np.array([1, 45, 85]), theta))
print ('For a student with scores 45 and 85, we predict an admission ' \
         'probability of %f\n' % prob)
print ('Expected value: 0.775 +/- 0.002\n\n')
# Compute accuracy on our training set
from predict import predict
p = predict(theta, X)
p = p.reshape((m, 1))
ans = np.mean(np.double(p == y)) * 100
print ('Train Accuracy: %f\n'% ans)
print ('Expected accuracy (approx): 89.0\n')
print ('\n')
