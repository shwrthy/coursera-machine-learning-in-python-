import numpy as np
import matplotlib.pyplot as plt
theta = np.array([1.27271026,  0.62529965,  1.18111687, -2.019874  , -0.9174319 ,
       -1.43166929,  0.12393227, -0.36553119, -0.35725405, -0.17516292,
       -1.4581701 , -0.05098417, -0.61558558, -0.27469165, -1.19271298,
       -0.24217841, -0.20603302, -0.04466178, -0.27778947, -0.29539514,
       -0.45645983, -1.04319154,  0.02779373, -0.29244865,  0.01555759,
       -0.32742404, -0.14389149, -0.92467488])
from mapFeature import mapFeature
u = np.linspace(-1, 1.5, 50)
v = np.linspace(-1, 1.5, 50)
z = np.zeros((len(u), len(v)))
for i in range(len(u)):
    for j in range(len(v)):
        z[i][j] = np.dot(mapFeature(np.array([u[i]]), np.array([v[j]])), theta)
z = z.T # important to transpose z before calling contour
plt.contour(u, v, z, label = "db")