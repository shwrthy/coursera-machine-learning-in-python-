import numpy as np
import matplotlib.pyplot as plt
from plotData import plotData
from mapFeature import mapFeature

def plotDecisionBoundary(theta, X, y):
# Plots the data points X and y into a new figure with
# the decision boundary defined by theta
    plotData(X[:,1:], y) # Plot Data
    if X.shape[1] <= 3:
        # Only need 2 points to define a line, so choose two endpoints
        plot_x = [min(X[:,1]) - 2,  max(X[:,1]) + 2] 
        # Calculate the decision boundary line
        plot_y = np.multiply((-1/theta[2]), (np.multiply(plot_x, \
                             theta[1]) + theta[0]))
        plt.plot(plot_x, plot_y)
    else :
        # Here is the grid range
        u = np.linspace(-1, 1.5, 50)
        v = np.linspace(-1, 1.5, 50)
        z = np.zeros((len(u), len(v)))
        for i in range(len(u)):
            for j in range(len(v)):
                z[i][j] = np.dot(mapFeature(np.array([u[i]]), np.array([v[j]])), theta)
        z = z.T # important to transpose z before calling contour
        plt.contour (u, v, z)
        
