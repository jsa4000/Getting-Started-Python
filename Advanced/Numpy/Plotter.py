from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def GenerateScatterPlot(x, y):
     plt.scatter(x,y)

def Generate3DPlot(x,y, func):
    fig = plt.figure()
    ax = Axes3D(fig)
    X, Y = np.meshgrid(x, y)
    Z = func(X,Y)
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.cm.hot)
    ax.contourf(X, Y, Z, zdir='z', offset=-2, cmap=plt.cm.hot)
    ax.set_zlim(-2, 2)
    plt.show()
    
def GeneratContour(inputs, f):
    x = inputs[:,0]
    y = inputs[:,1]
    X,Y = np.meshgrid(x, y)

    plt.axes([0.025, 0.025, 0.95, 0.95])

    plt.contourf(X, Y, f(inputs), 8, alpha=.75, cmap=plt.cm.hot)
    C = plt.contour(X, Y, f(inputs), 8, colors='black', linewidth=.5)
    plt.clabel(C, inline=1, fontsize=10)

    plt.xticks(())
    plt.yticks(())
    plt.show()
