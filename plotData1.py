import matplotlib.pyplot as plt
import numpy as np

def plotData1 (X, y, theta):
    plt.figure (2)
    m = np.size(X, 0)
    X1 = X[:, 1]
    X1 = X1.reshape((m, 1))
    plt.scatter (X1, y, color = 'red', marker = 'x')
    plt.plot(X1, np.dot (X, theta), color = 'blue')
    plt.xlabel ('Population of city in 10,000s')
    plt.ylabel ('Profit in $10,000s')
    plt.show ()
    return
