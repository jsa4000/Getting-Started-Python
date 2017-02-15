import numpy as np

def sigmoid(x, Derivative=False):
    if not Derivative:
        return 1.0 / (1.0 + np.exp(-x))
    else:
        out = sigmoid(x)
        return out * (1.0 - out)
    
def linear(x, Derivative=False):
    if not Derivative:
        return x
    else:
        return 1.0
    
def gaussian(x, Derivative=False):
    if not Derivative:
        return np.exp(-x**2)
    else:
        return -2*x*np.exp(-x**2)
    
def tanh(x, Derivative=False):
    if not Derivative:
        return np.tanh(x)
    else:
        return 1.0 - np.tanh(x)**2
    
def truncLinear(x, Derivative=False):
    if not Derivative:
        y = x.copy()
        y[y < 0] = 0
        return y
    else:
        return 1.0

def leastSquares(x, y):
     return np.sum((x - y)**2 ) * 0.5 