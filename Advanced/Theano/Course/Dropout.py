import theano
import theano.tensor as T
import numpy as np

# from theano.tensor.shared_randomstreams import RandomStreams
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
srng = RandomStreams()

# Let's investigate  what dropout is actually doing to the neuron's activation and how affect
# Dropout is commonly used for training optimization and for solving overfitting problems in predictions.

# Declare the Matric with the X
x = T.matrix()
p = T.scalar()  # For indicate a probability for the dropout to success.

def dropout2(x, p):
    #if p.gt(p,0):
    #   x = T.switch(srng.binomial(size=x.shape,p=p),x,0) # Rhis ecuation is the 
    #return x 
    return theano.ifelse.ifelse(T.gt(p,0), T.switch(srng.binomial(size=x.shape,p=p),x,0), x)

def binFilter(x, p):
    retain_prob = 1 - p
    x *= srng.binomial(x.shape, p=retain_prob, dtype=theano.config.floatX) # Generating a random mask to take random activations
    x /= retain_prob
    return x

#See how to compare tensor varaiable with something else.
#   http://deeplearning.net/software/theano/library/tensor/basic.html?highlight=eq#theano.tensor.eq
def dropout(x, p):
    #if p.get_value() > 0:   # Only shared theano variables
    #if p.eval() > 0:        #theano variables outside the scope of theano's definition graph
    #if T.gt(p,0):
    #    retain_prob = 1 - p
    #    x *= srng.binomial(x.shape, p=retain_prob, dtype=theano.config.floatX) # Generating a random mask to take random activations
    #    x /= retain_prob
    #return x

    return theano.ifelse.ifelse(T.gt(p,0), binFilter(x,p), x)


# Let define the function for the theano graph
y = dropout2(x, p)

# compile the graph defined
f = theano.function ([x,p], y)

# Now that we have the functions, variables, etc, compiled now lots compute numerical data

#Declare a layer with 4 units and 10 observations.
a = np.random.randn(10, 4)
print (a)

prob = 0.5 # -> p = 0 means no dropout and no testing

result = f(a,prob)
print (result)

result = f(a,0)
print (result)

#Before the dropout
#[[-0.34291894  0.02748842 -0.39793516 -0.23241725]
# [ 0.28451404  0.6542604   0.05229011 -0.61434242]
# [ 0.69091151 -0.89775655 -0.07503274 -0.06970099]
# [-2.67228931  0.0566169  -0.23462554 -0.22423444]
# [ 0.38121783  0.99491439 -2.19185316 -0.51483635]
# [-1.09021127 -0.15947066  1.64725148 -0.79415808]
# [-0.61186306  0.05326349  2.09297418 -1.03384856]
# [ 0.82004517  1.55167871  0.23017513 -0.03918956]
# [ 1.29774503  0.11552674 -0.23803975 -0.79208805]
# [-2.12120671 -1.64214131 -0.54889005 -0.35453835]]


#After the dropout
#[[ 0.          0.02748842  0.          0.        ]
# [ 0.28451404  0.          0.         -0.61434242]
# [ 0.          0.         -0.07503274  0.        ]
# [ 0.          0.          0.          0.        ]
# [ 0.38121783  0.          0.         -0.51483635]
# [-1.09021127 -0.15947066  1.64725148  0.        ]
# [ 0.          0.          0.          0.        ]
# [ 0.82004517  1.55167871  0.23017513  0.        ]
# [ 1.29774503  0.         -0.23803975 -0.79208805]
# [-2.12120671  0.          0.         -0.35453835]]



##############3
## NUMPY  ##
##############

a = np.random.randn(10, 4)
print (a)


n, p = 1, .5  # number of trials, probability of each trial
s = np.random.binomial(n,p, size = a.shape)
# In binomial if n = 1 then it will take values from 0 to 1
# This is also called Bernoulli experiment
#    If n = 5 then it will take values from 0 to 5
# P is the probabiliy to be 0 or 1 in this case
#[[0 1 0 1]
# [1 0 1 0]
# [0 0 0 1]
# [0 0 0 0]
# [1 1 0 1]
# [0 0 0 0]
# [1 0 0 1]
# [1 1 1 1]
# [0 0 1 1]
# [1 1 0 1]]

print (s)

npdropout = a * s # element-wise matrix multiplication
print (npdropout)

#[[ 0.          0.0583958   0.          0.47631572]
# [-0.84208395 -0.         -1.23688809 -0.        ]
# [-0.          0.         -0.          0.04377968]
# [ 0.         -0.          0.         -0.        ]
# [-0.9212772   0.22482932  0.         -0.98199701]
# [-0.         -0.         -0.         -0.        ]
# [ 0.16239118 -0.         -0.          0.21921785]
# [ 0.40007024  0.02412758 -0.23827536 -1.84294641]
# [ 0.          0.         -0.57979416 -1.16817245]
# [-0.01107707 -1.94619225  0.          0.8506062 ]]