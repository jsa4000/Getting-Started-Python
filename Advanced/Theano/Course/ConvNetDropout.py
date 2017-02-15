import theano
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt

# This import are implicit for convolutional neural network.
from theano.tensor.nnet.conv import conv2d
from theano.tensor.signal.downsample import max_pool_2d


#http://cs231n.github.io/convolutional-networks/
#http://deeplearning.net/software/theano/library/tensor/nnet/conv.html#theano.tensor.nnet.conv2d
#http://deeplearning.net/tutorial/lenet.html
#http://www.wildml.com/2015/11/understanding-convolutional-neural-networks-for-nlp/

# from theano.tensor.shared_randomstreams import RandomStreams
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
srng = RandomStreams()

import load
# load data
x_train, t_train, x_test, t_test= load.cifar10(dtype=theano.config.floatX)
# Return an array with the arguments (index) with the maximun value.
labels_test= np.argmax(t_test, axis=1) 

# reshape data to convert them into 32x32 pixels data. 
x_train= x_train.reshape((x_train.shape[0], 1, 32, 32))  # No RGB information
x_test= x_test.reshape((x_test.shape[0], 1, 32, 32))

# define symbolic Theanovariables
x = T.tensor4()
t = T.matrix()

# define model: neural network
def floatX(x):
    return np.asarray(x, dtype=theano.config.floatX)

def init_weights(shape):
    return theano.shared(floatX(np.random.randn(*shape) * 0.1))

def momentum(cost, params, learning_rate, momentum):
    grads = theano.grad(cost, params)
    updates = []
    for p, g in zip(params, grads):
        mparam_i= theano.shared(np.zeros(p.get_value().shape, dtype=theano.config.floatX))
        v = momentum * mparam_i-learning_rate* g
        updates.append((mparam_i, v))
        updates.append((p, p + v))
    return updates

# Dropout is essencialy performed because the overfitting.
# Drop out basicalle use some funcionalities and percentage for the dropout in order to switch off some activations previously computed.
# So activations = dropout (activations, 0.5) -> Thi swill return the activations array but with some filtering and random values.

# Must be said that dropout must only be performed with training not when predictions. p = 0 then dropout won't be performed
#hidden1 = T.nnet.relu(T.dot(X,W_input_to_hidden1) + b_hidden1)
#if training:
#	hidden1 = T.switch(srng.binomial(size=hidden1.shape,p=0.5),hidden1,0)
#else:
#	hidden1 = 0.5 * hidden1
def dropout(x, p=0.):
    if p > 0:
        retain_prob = 1 - p
        x *= srng.binomial(x.shape, p=retain_prob, dtype=theano.config.floatX) # Generating a random mask to take random activations
        x /= retain_prob
    return x


# Basically now the model will be defined and the forward propagation will be symbolicy specified.
def model(x, w_c1, b_c1, w_c2, b_c2, w_h3, b_h3, w_o, b_o):
    # dimshuffle: is like reshape in numpy. Where x means new col/row and 0 the current position of matrix[0,]
    # See more documentation about it
    
    # First convolutional layer, between x and w_c1. A bias will be added to the result 
    c1 = T.maximum(0, conv2d(x, w_c1) + b_c1.dimshuffle('x', 0, 'x', 'x')) # RELU (CONV)
    # Max pool for the previous convolution layer with shape (3,3) -> DOWNSAMPLE
    p1 = max_pool_2d(c1, (3, 3))

    # Second convolutional layer, between last pool layer and w_c2. Also a bias will be added to the result 
    c2 = T.maximum(0, conv2d(p1, w_c2) + b_c2.dimshuffle('x', 0, 'x', 'x')) # RELU (CONV)
    # Max pool for the previous convolution layer with shape (2,2) -> Smaller that the first one -A
    p2 = max_pool_2d(c2, (2, 2))

    # Fully connected layer -> Also called Dense Layer
    p2_flat = p2.flatten(2) # flatter with two dimension (1000, 32*32) -> similar to the inputs when cifar was loaded

    p2_flat = dropout(p2_flat, p=p) # Add dropout

    h3 = T.maximum(0, T.dot(p2_flat, w_h3) + b_h3) # RELU (linear regression)

    # Last layer. A Softmax function will be computed using the previous output from the fully connected layer
    p_y_given_x= T.nnet.softmax(T.dot(h3, w_o) + b_o)

    # Finally return the value (0,1) because the sofmax function
    return p_y_given_x

# Creation of the Wigths and Biases for each layer

w_c1 = init_weights((4, 1, 3, 3)) # 4 kernels with 3x3 size
b_c1 = init_weights((4,))

w_c2 = init_weights((8, 4, 3, 3))  # 8 kernels with 2x2 size, for each previous extracted feature
b_c2 = init_weights((8,))

# 4 * 4 are the size of the image after the max pooling and 8 the total of features maps
w_h3 = init_weights((8 * 4 * 4, 100)) # Size of the final feauters extracted (using previous sizes). Size of the fully connected layer 
b_h3 = init_weights((100,))

w_o= init_weights((100, 10)) # Notmal weight of a NN, using the last unit's laters with the units for the next one. 
b_o= init_weights((10,)) #The sum of the bias (1D) must coincide with the size of the colums (M) of the final matrix  (NxM)

params = [w_c1, b_c1, w_c2, b_c2, w_h3, b_h3, w_o, b_o]

p_y_given_x_train = model(x, *params, p=0.5)
p_y_given_x_test = model(x, *params, p=0.0)   # Train doesn't do dropout -> Forward
y_train = T.argmax(p_y_given_x_train, axis=1)
y_test = T.argmax(p_y_given_x_test, axis=1)

cost_train = T.mean(T.nnet.categorical_crossentropy(p_y_given_x_train, t))

updates = momentum(cost_train, params, learning_rate=0.01, momentum=0.9)

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

import plot_utils
plot_utils.visualize_grid(w_c1.get_value())
plot_utils.visualize_grid(w_c2.get_value()[:, 0])
plot_utils.visualize_grid(w_c2.get_value()[:, 1])