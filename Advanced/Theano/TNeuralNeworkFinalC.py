import load
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
  #   Relu al contrario que Sigmoid puede tomar valorees desde 0 a infinito
  #   http://cs231n.github.io/neural-networks-1/
  # Softmax function, in neural network simulations, is often implemented at the final layer of a 
  #   network used for classification. Such networks are then trained under a log loss (or cross-entropy) 
  #   regime, giving a non-linear variant of multinomial logistic regression.
  #   Basically this function is used for probability due the range of the possible values go from 0 to 1. So it can be used
  #   in termns of probability P(y == j|z) and in conjunction with cost function based on cross-entropy for matching result.
  #      P(y == j|z)  = softmax (z) = np.exp(z) / np.sum(np.exp(z))
  #   In the field of reinforcement learning, a softmax function can be used to convert values into action probabilities
  

# Another thing that it's very connected to the final outputs are the way to evaluate the prediction. In this case depending
# on the final regression or classifier we must use a cost function which perform the error of the model depending on the 
# training data.
  
# After chooosing the model and the cost function we need to choose the back-propagation method in order to update teh  weights:
# Deffierent choices could be done: Gradient descent, Momentum, SGD (Stochastic gradient descent), Batch, min-batch. 
# This will depend on the level of optimization you want to use in your model. 

#In theano, variables and functions must be declared previously in a symbolic way.
# After compile the functions and variables, a numerical computation could be performend.
# This is to accelerate the computation time and get performances better than libraries like numpy.
# Three methods could be used using python: Numpy, numpy + Theano (shared variables) and only theano.

# Now we are going to define the inputs... array of bytes. The images are plain arrays in 1D dimension. 32 x 32 = 1024 inputs + bias

# Before defining the Symbolic implementation for theano. A shape and the functions for the model will be required

def linear(x):
    return x

def ReLu(x):
    return T.maximum(0,x)

shape = (32*32,100,100,10)
	
#It's important to match transfer functions (sigmoid, relu, etc.) to cost functions (softmax, least squares, etc.) 
#to avoid creating additional local optima. A softmax is meant to scale sigmoidal units to be interpreted as a multi-class
# probability. Try using least-squares with relu and see if you get better results?

functions = (None, ReLu, ReLu, T.nnet.softmax)

# Symbolic Definition for the Theano implementation

# 1. Define the Inputs that the model will take in order to train the Network.
x = T.matrix() # Matrix because if will be multiple columns and multiple row with multiple observations- Why I must use 4D matrix instead
t = T.matrix() # Our training data based on observation. Expected outputs based in the previous inputs.

# 2. Define the Weights inbetween the layers

def layerCount (shape):
    return len(shape)

def floatX(x):
    return np.asarray(x, theano.config.floatX) 

def init_weights (shape, mode = "R"):
    if mode == "Z":
        return theano.shared(floatX(np.zeros(shape)))
    elif mode == "G":
        return theano.shared(floatX(np.random.normal(scale=0.01, size = shape)))
    else:
        return theano.shared(floatX(np.random.randn(*shape) * 0.1))  

def init_biases (n):
    return theano.shared(floatX(np.ones((n,))))
    
# Define BIAS term for the weights (also it will be initializated with random numbers)
usebias = False

# Check whether the bias is used (Don't know if the bias can be vectorized inside theano...)
biases = []
if (usebias):
    for m, n in zip(shape,shape[1:]):
        biases.append(init_biases((n)))
        
weights = []
for m, n in zip(shape,shape[1:]):
    weights.append(init_weights((m, n)))
   
# Define the Back propagation algorithm and the updates to be done by the gradient
# For the back propagation algorithm I need: cost function (which error will be backpropagated), weights and training rate
# If mometum wants to be added momentum speed and previous weights are also needed
def backpropagation(cost, weights, training_rate= 0.25, momentum = 0.15 ):
    grads = T.grad(cost,weights)
    updates = []
    for w, g in zip(weights, grads):
        # Create a temporal variable for the previous weight for the momentum
        pw = init_weights(w.get_value().shape,"Z")
        # Add the training rate and the momentum to the delta
        delta = momentum * pw - g * training_rate 
        updates.append((pw, delta))
        updates.append((w,w + delta))
    return updates
   
# Now define the model used
# For the model, depending on the layer we want to use diferent shapes and functions
# For the activation for each layer we will compute the standard linear function w*x + bias, 
# and finally use the activation function for non-linearity, except for the final layer
def model(input_layer, shape , functions, weights, biases):
    activations = []
    for layer in range(1,layerCount(shape)):
        bias = 0
        if (len(biases) != 0):
            bias =  biases[layer-1]

        if (layer==1):
            activations.append(functions[layer](T.dot(x,weights[layer-1]) + bias))
        else:
            activations.append(functions[layer](T.dot(activations[-1],weights[layer-1])  + bias))
    return activations[-1]

# creates the model
y = model(x, shape, functions, weights, biases)

# Define the cost function. In this case we will use back-propagation algorithm, by using gradient descent with least squares cost
#cost = T.mean((y - t)**2)
cost = T.mean(T.nnet.categorical_crossentropy(y, t))

# Create the back propagation algorithm with the updates aapplied to the model
updates = backpropagation(cost, weights, training_rate = 0.25,  momentum = 0.15)

# Finaly creates the function to train and predict using the definition already done
train = theano.function([x,t], cost, updates = updates)

 #np.argmax(y, axis=1) -> This is to return the neuron with highest activation
predict = theano.function ([x],np.argmax(y, axis=1))

# Now compute the numerical stage in order to train the network

# load data
x_train, t_train, x_test, t_test= load.cifar10(dtype=theano.config.floatX)
labels_test= np.argmax(t_test, axis=1)

# train model
batch_size= 50
for i in range(100):
    print "iteration %d" % (i+ 1)
    for start in range(0, len(x_train), batch_size):
        x_batch= x_train[start:start+ batch_size]
        t_batch= t_train[start:start+ batch_size]
        cost = train(x_batch, t_batch)
    predictions_test= predict(x_test)
    accuracy = np.mean(predictions_test== labels_test)
    print "accuracy: %.5f" % accuracy
    print
