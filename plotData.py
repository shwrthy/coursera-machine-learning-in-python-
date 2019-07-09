# Instructions: Plot the training data into a figure using the 
#               "figure" and "plot" commands. Set the axes labels using
#               the "xlabel" and "ylabel" commands. Assume the 
#               population and revenue data have been passed in
#               as the x and y arguments of this function.

import matplotlib.pyplot as plt

def plotData (X, y):
    plt.figure (1)
    plt.scatter (X, y, color = 'red', marker = 'x')
    plt.xlabel ('Population of city in 10,000s')
    plt.ylabel ('Profit in $10,000s')
    plt.show ()
    return
