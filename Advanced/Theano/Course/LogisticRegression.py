import load
import theano
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt
import theano.tensor.nnet as theano_nnet

# You can use it to regress probabilities
# Logistic regression belongs to the GLM family of models
#   https://en.wikipedia.org/wiki/Generalized_linear_model
    #Fitting: 
        #- The maximum likelihood estimates can be found using an iteratively reweighted least squares algorithm using either
        #a Newton–Raphson method with updates of the form:
        #- Bayesian methodsIn general, the posterior distribution cannot be found in closed form and so must be approximated, 
        #usually using Laplace approximations or some type of Markov chain Monte Carlo method such as Gibbs sampling. 

# Its related to the LOGISTIC distribution, which has an S-shaped curve.


# load data
x_train, t_train, x_test, t_test= load.cifar10(dtype=theano.config.floatX)
labels_test= np.argmax(t_test, axis=1) # if the test x1 = 0 0 1 0, then return the argument with the maximun, in this case 2
# visualize data
#plt.imshow(x_train[0].reshape(32, 32), cmap=plt.cm.gray)
#plt.show()

# Defin symbolic variables in Theano

x = T.matrix()
t = T.matrix()

# Define a function to convert a float matrix into a theano shared np array
def floatX (x):
    return np.asarray(x, theano.config.floatX)
 
# Init the weights 
def init_weights(shape):
    return theano.shared(floatX(np.random.rand(*shape) * 0.1)) # the asterisk in shape transfor the tuple into different values (remove the tuple)

# Define the model 
# The model defined if softmax -> exp(x*w) -> If normalized Softmax -> Sum (x*w)
def model(x,w):
    return T.nnet.softmax(T.dot(x,w)) # This is the function used within the model softmax for all x and weights passed by parameters.

# Create the weights (32*32, 10) size oof the images and the number of features
w = init_weights ((32*32, 10))  # In logistic, the columns of the weight must coincide with the outputs size

# Create the symbolic function tobe passed into the theano's function. Since it's the max we will use the argmax value encountered
p_y_given_x= model(x, w)
y = T.argmax(p_y_given_x, axis=1)

# Create the evaluation function or cost function
cost = T.mean(T.nnet.categorical_crossentropy(p_y_given_x, t))

#Define the update for the weights and gradient descent
g = T.grad(cost, w) # Create the gradient descent using the defined cost function and the weights
updates = [(w, w - g * 0.001)] # DEfine the update function for the weights. The BIAS used will be 0.001

# Finally create the function in Theano with all symolic parameters already defined. Compile theano functions
train = theano.function ([x,t],cost, updates = updates)
predict = theano.function([x], y)

# train model
batch_size= 50
for i in range(100):
    print "iteration %d" % (i+ 1)
    for start in range(0, len(x_train), batch_size):
        x_batch= x_train[start:start+ batch_size]
        t_batch= t_train[start:start+ batch_size]
        cost = train(x_batch, t_batch)
        print "cost: %.5f" % cost
    predictions_test= predict(x_test)
    accuracy = np.mean(predictions_test== labels_test)
    print "accuracy: %.5f" % accuracy
    print


