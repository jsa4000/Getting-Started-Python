"""
THEANO

This is a very basic explanation explaining the basic functionality in Theano
This is not going to be intended to exaplain all the functionality that Theano
has since it's very complex.

This complexity includes function to compute:
    grad: to compute the derivatives of the functions (gradient)
    scan: to iterate over elements inthe graph
"""
import time
import numpy as np
import theano
import theano.tensor as T
# pylint: disable=invalid-name
# pylint: disable=E1101

# Theano is a symbolic language that generates internal graphs for the overall computation.
# This graph is useful for parallel computation, differenciation and integration since
# all the different functions, constant and variables are stored.
#
#
# Finally you can declare differen fucntions using, starting point, inputs, outputs, etc..
# This graph need to be compiled in theano gramwork. It's all compiled automatically and
# it generates the code (c/c++) necessary to run the graph.
#
# Theano's main features are:
# - GPU performances. By using Nvidea CUDA cores
# - Integration with Numpy
# - High speed compared to other frameworks.
#
# Theano's bad points are:
# - TensorFlow.
# - Low in compilation time.
# - Lack of in-built function for Machine learning


# First at all, it's important to declare first all the symbolic variables.
# Symbolic variables are the variables that will be used in the computation graph

#You can cdeclare Tensor. Tensor could be a Vector or a Matrix.abs
X = T.matrix("X")
V = T.vector("V")


# You nned to specify the length of the vector.abs
# This is bacuse, the Shape it's needed for vectorization
#V2 = T.vector3("V2")

S = T.scalar("S")
C = T.constant(3)


# NOTE: You cannot access to the symbolic variables until the last very moment when the
# function has been computed.

#Another Thing that it's needed it's to define the operations or functions that will be used.abs
# It can be nested as many operations as needed. However the compilation will be must slower and
# theano computation time aslso slower.
def multiplication(x, y):
    """
    This function will return the dot product of two matrix (parallel)
    It's recomended to use the function provided in theano for best performances.

    """
    return T.dot(y, x)

# Also you can add shared variables. This variables can be initialized. However the content
# cannot be accessed since the modifications are not going to be updated

#There are some functions that are useful to initialize and create some variables
def floatX(x):
    """
    This function return an numpy array with the configured float in theano
    Depending on if it's used GPU or CPU, etc.. this could change.
    """
    return np.asarray(x, theano.config.floatX)

def init_variable(shape):
    """
    This method will return a shared variable with the given shape
    The initialization will be done  by usineg a uniform distribution
    However this initialization could be any.
    """
    return theano.shared(floatX(np.random.normal(scale=0.01, size=shape)))

## Now lets create a shared variable with the given shape
shared_variable = init_variable((2, 2))

# Now that we have the symbolic variables, functions and shared variables.
# We can finally create the function that will be used for theano to
# create the graph

inputs = [[2], [3]]

#Lets print the inputs and the initializartion weights
print(inputs)
# if you try to print the shared variable it will give you a tensortype type
#This is the varible we have initialized previously
print(shared_variable)
# Also if you need to print the T variable you will get the M name, if not
# you will get a Tensot type variable.
print(X)

# X Will represent the "inputs"" and Y the outputs, that will return the
# graph using the model defined.

# Generate a function with the output you will get.
Y = multiplication(X, shared_variable)

# Now lets compile the function with theano
my_func = theano.function([X], Y)

# Now lets call the function with the desired inputs already created
result = my_func(inputs)
print(result)


# Let's add a print function at the end so we will now when the function ahs been ended
print("END")

# Theano are able to do more advanced studd. But this example show an overview of
# its basic functionality.