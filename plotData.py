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
    plt.scatter (x1[:,0], x1[:,1], label = 'Admitted', color = 'black', marker = '+')
    plt.scatter (x2[:,0], x2[:,1], label = 'Not admitted', color = 'yellow', marker = 'o')
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')
    plt.legend () # Automatic detection of elements to be shown in the legend
    
