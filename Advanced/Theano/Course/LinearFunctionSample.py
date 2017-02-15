import theano
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt

# Some useful links.
#https://www.udacity.com/course/deep-learning--ud730?utm_campaign=chrome_series_machinelearning_063016&utm_source=gdev&utm_medium=yt-desc
#http://colah.github.io/posts/2015-09-Visual-Information/
#http://colah.github.io/posts/2015-08-Backprop/
#https://research.googleblog.com/2016/03/train-your-own-image-classifier-with.html
#https://www.youtube.com/watch?v=1RphLzpQiJY
#https://www.youtube.com/watch?v=BFdMrDOx_CM

#http://scikit-learn.org/stable/tutorial/statistical_inference/supervised_learning.html

#plt.ion()
#plt.show()

# create artificial training data
trainingRate = 0.01 # Training ratio
x_train= np.linspace(-1, 1, 101)
t_train= 2 * x_train+ np.random.randn(*x_train.shape) * 0.33
# plot data
plt.scatter(x_train, t_train)
plt.show()

#Now define the symbolic variables (what if vectorize where x are multiple features)

x = T.scalar()  # Scalars with the training inputs
t = T.scalar() # Scalars with the result of the training data

# Define the model that will be used
def model(x, w):
    return x * w

## Define the model that will be used
#def model(x, w):
#    return x * w + bias

# Define wegiths, gradient, cost function, update for the weight when training the model.
w = theano.shared(0.0) # Only 1 weight because 1 feature y = ax + b
y = model (x,w)  # symboloc definition for the model given the weights

cost = T.mean((t - y)**2) # Cost function to evaluate the model

g = T.grad(cost, w) # Define the gradient descent as the derivate of the cost depending on w
updates = [(w,w - g * trainingRate)]# Formule to define the update of weights

#Create the function to model the entire thing.
train = theano.function([x, t], cost, updates = updates )

# train model
for i in range(20):
    print "iteration %d" % (i+ 1)
    for x, t in zip(x_train, t_train):
        train(x, t)
    print "w = %.8f" % w.get_value()
    print

# plot fitted line
#plt.plot(x_train, w.get_value() * x_train)
plt.plot(x_train, model( x_train, w.get_value()))
plt.show()

