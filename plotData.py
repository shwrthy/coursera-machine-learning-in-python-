import numpy as np
import matplotlib.pyplot as plt

# Instructions: Plot the positive and negative examples on a
#               2D plot, using the option 'k+' for the positive
#               examples and 'ko' for the negative examples.

def plotData(X, y):
    m, n = X.shape
    y = y.reshape((m, ))
    x1 = X[y == 1,:]
    x2 = X[y == 0,:]
    plt.figure ()
    plt.scatter (x1[:,0], x1[:,1], color = 'black', marker = '+')
    plt.scatter (x2[:,0], x2[:,1], color = 'yellow', marker = 'o')
    
