import theano
import theano.typed_list
import theano.tensor as T
import numpy as np

# Define the symbolic variables
x = T.matrix()
n = T.iscalar()

def floatX (x):
    return np.asarray(x,theano.config.floatX)

shape = (3,4)
t = theano.shared(floatX(np.random.rand(*shape)))

#Defin the function
def getVector(matrix, index):
    return matrix[index]

#Defin the function
def getVector2(index):
    return t[index]

#Create the function
y = getVector(x,n)
y2 = getVector2(n)

#Compile the function
f = theano.function([x,n],y)
f2 = theano.function([n],y2)

# GEt the numeric computation
inputMatrix = np.random.rand(3,4)
print(inputMatrix)
index = 1

vector = f(inputMatrix,index)
print (vector)

vector = f2(index)
print (vector)


#Shared variable helps in simplifying the operations over a pre-defined variable. An example to @danien-renshaw 's answer, suppose we want to add two matrix, let's say a and b, where the value of b matrix will remain constant throughout the lifetime of the program, we can have the b matrix as the shared variable and do the required operation.

#Code without using shared variable:

#a = theano.tensor.matrix('a')
#b = theano.tensor.matrix('b')
#c = a + b
#f = theano.function(inputs = [a, b], outputs = [c])
#output = f([[1, 2, 3, 4]], [[5, 5, 6, 7]])

#Code using shared variable :

#a = theano.tensor.matrix('a')
#b = theano.tensor.shared( numpy.array([[5, 6, 7, 8]]))
#c = a + b
#f = theano.function(inputs = [a], outputs = [c])
#output = f([[1, 2, 3, 4]])

