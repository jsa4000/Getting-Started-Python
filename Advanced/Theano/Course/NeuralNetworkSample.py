import theano
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as ds
import load

# load data
x_train, t_train, x_test, t_test= load.cifar10(dtype=theano.config.floatX)
labels_test= np.argmax(t_test, axis=1) # Gives the argument with  maximun value. eg. [0, 2, 1, 0] -> Returns 1 since value = 2 is the max arg
# visualize data

# Defin symbolic variables in Theano

x = T.matrix()
t = T.matrix()

# define model: neural network
def floatX(x):
    return np.asarray(x, dtype=theano.config.floatX)

def init_weights(shape):
    return theano.shared(floatX(np.random.randn(*shape) * 0.1))

#  Define the Back-propagation algoritm (gradient descent) to update the weights
def sgd(cost, params, learning_rate):
    # Define gradient descent on params
    grads = T.grad(cost, params)
    updates = []
    # For each weights (params) define the gradient to be computed
    for p, g in zip(params, grads):
        updates.append([p, p - g * learning_rate])
    return updates

# Define the model
def model(x, w_h, w_o):
    # Define the hidden lyers models
    h = T.maximum(0, T.dot(x, w_h)) # RELU 
    # DEfine the output layer model
    p_y_given_x= T.nnet.softmax(T.dot(h, w_o))
    return p_y_given_x

# Define the weights for hidden and output layers
w_h= init_weights((32 * 32, 100)) # Define 32*32 x 100 inputs and hidden layers
w_o= init_weights((100, 10)) # Define 100x10, where 100 hidden units and 10 outputs units (needed for the softmax)

# Define the final output layer using the model
p_y_given_x= model(x, w_h, w_o)
y = T.argmax(p_y_given_x, axis=1)

# Define the cost function
cost = T.mean(T.nnet.categorical_crossentropy(p_y_given_x, t))
params= [w_h, w_o]
updates = sgd(cost, params, learning_rate=0.01)

# compile theanofunctions
train = theano.function([x, t], cost, updates=updates)
predict = theano.function([x], y)

# train model
batch_size= 50
for i in range(50):
    print "iteration %d" % (i+ 1)
    for start in range(0, len(x_train), batch_size):
        x_batch= x_train[start:start+ batch_size]
        t_batch= t_train[start:start+ batch_size]
        cost = train(x_batch, t_batch)
    predictions_test= predict(x_test)
    accuracy = np.mean(predictions_test== labels_test)
    print "accuracy: %.5f" % accuracy
    print