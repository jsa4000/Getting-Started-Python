from matplotlib import pyplot as plt
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#http://www.scipy-lectures.org/intro/matplotlib/matplotlib.html#scatter-plots


def basic3DPlot01():
    fig = plt.figure()
    ax = Axes3D(fig)
    X = np.arange(-4, 4, 0.25)
    Y = np.arange(-4, 4, 0.25)
    X, Y = np.meshgrid(X, Y)
    R = np.sqrt(X ** 2 + Y ** 2)
    Z = np.sin(R)

    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.cm.hot)
    ax.contourf(X, Y, Z, zdir='z', offset=-2, cmap=plt.cm.hot)
    ax.set_zlim(-2, 2)

    plt.show()
    
def basicPlot01():
    # Define R for X. In this case the -pi to pi (256 values)
    X = np.linspace(-np.pi, np.pi, 256, endpoint=True)
    # Defin two functions to plot
    C, S = np.cos(X), np.sin(X)

    # Plot the first function within X values
    plt.plot(X, C)
    # Plot second function within X values
    plt.plot(X, S)

    plt.show()

def basicPlot02():
    # Define R for X. In this case the -pi to pi (256 values)
    X = np.linspace(-np.pi, np.pi, 256, endpoint=True)
    # Defin two functions to plot
    C, S = np.cos(X), np.sin(X)

    # Plot cosine with a blue continuous line of width 1 (pixels)
    plt.plot(X, C, color="blue", linewidth=1.0, linestyle="-")

    # Plot sine with a green continuous line of width 1 (pixels)
    plt.plot(X, S, color="green", linewidth=1.0, linestyle="-")

    # Set x limits
    plt.xlim(-4.0, 4.0)

    # Set x ticks
    plt.xticks(np.linspace(-4, 4, 9, endpoint=True))

    # Set y limits
    plt.ylim(-1.0, 1.0)

    # Set y ticks
    plt.yticks(np.linspace(-1, 1, 5, endpoint=True))

    # Show result on screen
    plt.show()

def basicScatter01():
    n = 1024
    X = np.random.normal(0,1,n)
    Y = np.random.normal(0,1,n)

    plt.scatter(X,Y)

def f(x, y):
    return (1 - x / 2 + x ** 5 + y ** 3) * np.exp(-x ** 2 -y ** 2)

def contours():
    n = 256
    x = np.linspace(-3, 3, n)
    y = np.linspace(-3, 3, n)
    X,Y = np.meshgrid(x, y)

    plt.axes([0.025, 0.025, 0.95, 0.95])

    plt.contourf(X, Y, f(X, Y), 8, alpha=.75, cmap=plt.cm.hot)
    C = plt.contour(X, Y, f(X, Y), 8, colors='black', linewidth=.5)
    plt.clabel(C, inline=1, fontsize=10)

    plt.xticks(())
    plt.yticks(())
    plt.show()

basic3DPlot01()