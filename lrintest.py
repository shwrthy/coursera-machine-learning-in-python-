import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 4, 100)
data = pd.read_csv ('ex1data1.txt', header = None)
data = np.array (data)
X = data [:, 0]
y = data [:, 1]
m = np.size (X, 0) #number of training examples
X = X.reshape ((m, 1))
y = y.reshape ((m, 1))
X = np.c_ [np.ones((m, 1)), X] # Add a column of ones to x

from computeCost import computeCost

J_vals = np.zeros((len(theta0_vals), len(theta1_vals)))

# Fill out J_vals
for i in range(len(theta0_vals)):
    for j in range(len(theta1_vals)):
        t = np.array([theta0_vals[i], theta1_vals[j]])
        t = t.reshape((2, 1))
        J_vals[i,j] = computeCost(X, y, t)
