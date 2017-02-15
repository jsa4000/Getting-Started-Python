import theano
from theano import tensor as T
import numpy as np
from matplotlib import pyplot as plt
import sklearn.datasets as ds

# This class will create a Neural Network using theano.
# The model will depend on the final result to obtain, model and inputs being used: images, numbers, binary digits, etc.. 
# The result obtained (predicted) coukd be computed using logistics functions (probability), linear regression, etc..
# in order to get a classification or scalar value at the end. For image recognition classifiers will be almost required.
  
  # Some diferent activations could be Sigmoid, ReLu, softmax (outer layers), tanh, etc...
  # ReLu is basically switch(x<0, 0, x) -> Only take the possitive values greater than 0 or lambda x: T.maximum(0,x)
   # http://cs231n.github.io/neural-networks-1/

# Another thing that it's very connected to the final outputs are the way to evaluate the prediction. In this case depending
# on the final regression or classifier we must use a cost function which perform the error of the model depending on the 
# training data.
  
# After chooosing the model and the cost function we need to choose the back-propagation method in order to update teh  weights:
# Deffierent choices could be done: Gradient descent, Momentum, SGD (Stochastic gradient descent). This will depend on the 
# level of optimization you want to use in your model. 

#In theano, variables and functions must be declared previously in a symbolic way.
# After compile the functions and variables, a numerical computation could be performend.
# This is to accelerate the computation time and get performances better than libraries like numpy.
# Three methods could be used using python: Numpy, numpy + Theano (shared variables) and only theano.

# Now we are going to define the inputs... array of bytes. The images are plain arrays in 1D dimension. 32 x 32 = 1024 inputs + bias


# Before defining the Symbolic implementation for theano. A shape and the functions for the model will be required

def linear(x):
    return x

shape = (2,5,1)
functions = (None, T.nnet.sigmoid, linear)

# Symbolic Definition for the Theano implementation

# 1. Define the Inputs that the model will take in order to train the Network.
x = T.matrix() # Matrix because if will be multiple columns and multiple row with multiple observations- Why I must use 4D matrix instead
t = T.matrix() # Our training data based on observation. Expected outputs based in the previous inputs.

# 2. Define the Weights inbetween the layers

def layerCount (shape):
    return len(shape)

def floatX(x):
    return np.asarray(x, theano.config.floatX) 

def init_weights (shape):
    #return theano.shared(floatX(np.random.randn(*shape) * 0.2))  #np.random.normal(scale=0.01,
    return theano.shared(floatX(np.random.normal(scale=0.01, size = shape)))

# Define BIAS term for the weights (also it will be initializated with random numbers)
bias = 0

weights = []
for m, n in zip(shape,shape[1:]):
    weights.append(init_weights((m + bias, n + bias)))

#previousweights = []
#for m, n in zip(shape,shape[1:]):
#    weights.append(init_weights((m + bias, n + bias)))

# Define the Back propagation algorithm and the updates to be done by the gradient
# For the back propagation algorithm I need: cost function (which error will be backpropagated), weights and training rate
# If mometum wants to be added momentum speed and previous weights are also needed
def backpropagation(cost, weights, training_rate):
    grads = T.grad(cost,weights)
    updates = []
    for w, g in zip(weights, grads):
        updates.append((w,w - g * training_rate))
        #updates.append((w,w - g * training_rate + momentum * pw))
    return updates
   
# Now define the model used
# For the model, depending on the layer we want to use diferent shapes and functions
# For the activation for each layer we will compute the standard linear function w*x + bias, 
# and finally use the activation function for non-linearity, except for the final layer
def model(input_layer, shape , functions, weights):
    activations = []
    for layer in range(1,layerCount(shape)):
        if (layer==1):
            activations.append(functions[layer](T.dot(x,weights[layer-1])))
        else:
            activations.append(functions[layer](T.dot( activations[-1],weights[layer-1])))
    return activations[-1]

# creates the model
y = model(x, shape, functions, weights)

# Define the cost function. In this case we will use back-propagation algorithm, by using gradient descent with least squares cost
cost = T.mean((y - t)**2)

# Create the back propagation algorithm with the updates aapplied to the model
updates = backpropagation(cost, weights, training_rate = 0.25)

# Finaly creates the function to train and predict using the definition already done
train = theano.function([x,t], y, updates = updates)
predict = theano.function ([x],y)

# Now compute the numerical stage in order to train the network
data = ds.load_iris()
x_train = data["data"][:,:2]
t_train = data["target"]

# train model
maxError =  0.047
maxIterations = 100000

def getError(y,t):
    return np.mean((y - t)**2)

for iter in range(0, maxIterations):
    # Call update() -> Forward propagation + update weights and Bias
    results = train(x_train, t_train[:,None])
    error = getError(results, t_train[:,None])
    # Get the current Error (Const function) implemented by the trainer
    if iter % 5000 == 0 and iter > 0:
        print "cost = %.8f" % error
    if error <= maxError:
        print("Desired error reached. Iter: {0}".format(iter))
        break

output = predict(x_train[0][None,:])
error = t_train[0] - x_train[0]

# plot fitted line
#plt.plot(x_train, w.get_value() * x_train)
#plt.plot(x_train, model( x_train, w.get_value()))
#plt.show()
