import theano
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as ds

# Some useful links.
#https://www.udacity.com/course/deep-learning--ud730?utm_campaign=chrome_series_machinelearning_063016&utm_source=gdev&utm_medium=yt-desc
#http://colah.github.io/posts/2015-09-Visual-Information/
#http://colah.github.io/posts/2015-08-Backprop/
#https://research.googleblog.com/2016/03/train-your-own-image-classifier-with.html
#https://www.youtube.com/watch?v=1RphLzpQiJY
#https://www.youtube.com/watch?v=BFdMrDOx_CM

#http://scikit-learn.org/stable/tutorial/statistical_inference/supervised_learning.html

#1 Model used: Single Neouron with sum(X * W) + b to compute the prediction
#  ->  Cost function -> lest square since the result will be a scalar

#plt.ion()
#plt.show()

data = ds.load_iris()
x_train = data["data"][:,:2]
t_train = data["target"]

# create artificial training data
trainingRate = 0.01 # Training ratio
#x_train= np.linspace(-1, 1, 101)
#t_train= 2 * x_train+ np.random.randn(*x_train.shape) * 0.33
# plot data
#plt.scatter(x_train[:,0],x_train[:,1])
#plt.show()

#Now define the symbolic variables (what if vectorize where x are multiple features)

x = T.matrix()  # Scalars with the training inputs
t = T.matrix() # Scalars with the result of the training data

# Define the model that will be used
def model(x, w):
    return T.dot(x, w)

# Define a function to convert a float matrix into a theano shared np array
def floatX (x):
    return np.asarray(x, theano.config.floatX)
 
# Init the weights 
def init_weights(shape):
    return theano.shared(floatX(np.random.rand(*shape) * 0.1)) # the asterisk in shape transfor the tuple into different values (remove the tuple)


## Define the model that will be used
#def model(x, w):
#    return x * w + bias

# Define wegiths, gradient, cost function, update for the weight when training the model.
w = init_weights((x_train.shape[1], 1)) # Only 1 weight because 1 size output 

# This is the basic model used in liner -> w1x1 + w2x2 + w3x3 + w4*x4 + BIAS -> where x are features and w weights
y = model (x,w)  # symboloc definition for the model given the weights

cost = T.mean((t - y)**2) # Cost function to evaluate the model

g = T.grad(cost, w) # Define the gradient descent as the derivate of the cost depending on w
updates = [(w,w - g * trainingRate)]# Formule to define the update of weights

#Create the function to model the entire thing.
train = theano.function([x, t], cost, updates = updates )
predict = theano.function([x], y)

# train model

#for i in range(20):
#    print "iteration %d" % (i+ 1)
#    #for x, t in zip(x_train, t_train):
#    #    train(x.T, t)
#    thiscost = train(x_train, t_train[:,None])
#    #print "w = %.8f" % w.get_value()
#    print "cost = %.8f" % thiscost
#    print

maxError =  0.047
maxIterations = 100000

for iter in range(0, maxIterations):
    # Call update() -> Forward propagation + update weights and Bias
    error = train(x_train, t_train[:,None])
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

