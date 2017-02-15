import theano
import theano.tensor as T
import numpy as np


dimensions = 1
x = numpy.random.RandomState(1).rand(3, 2, 5, 5) # Max pool will take the last two indexes for the pooling


shape_x = np.asarray(x.shape)
o_shape = []
o_shape.extend(shape_x[range(dimensions - 1)]) #Get the (dimension - 1) values
o_shape.extend([np.prod(shape_x[(dimensions - 1):])]) #Final product from (dimensions - 1) to ndim
print tuple(o_shape)
print (x.reshape(tuple(o_shape)))


print ((shape_x[range(dimensions - 1)], np.prod(shape_x[(dimensions - 1):]) ))

shape_x[range(dimensions - 1)]
np.prod(shape_x[(dimensions - 1):])


def get_flattened_shape(x_shape, dimensions = 2):
    x_shape = np.asarray(x_shape)
    o_shape = []
    o_shape.extend(x_shape[range(dimensions - 1)])
    o_shape.extend([np.prod(x_shape[(dimensions - 1):])])
    return tuple(o_shape)

dimensions = 1
x = numpy.random.RandomState(1).rand(3, 2, 5, 5) # Max pool will take the last two indexes for the pooling
print (get_flattened_shape(x.shape, dimensions))



def flatten (w, dimension = 2):
    return x.flatten(dimension)

w = theano.shared( np.asarray(np.random.RandomState(1).rand(3, 2, 5, 5),dtype = theano.config.floatX) )


dimensions = 1
flattened = flatten (w,dimensions)
print (flattened)
print (flattened.shape)


d = T.constant(2)
y = flatten(x,t)

func = theano.function([w,d],y)

dimensions = 1
new = numpy.random.RandomState(1).rand(3, 2, 5, 5) # Max pool will take the last two indexes for the pooling

flattened = func (new,dimensions)
print (flattened)
print (flattened.shape)


w = theano.shared( np.asarray(np.random.RandomState(1).rand(3, 2, 5, 5),dtype = theano.config.floatX) )

def flatten (x):
    return x.flatten(2)

y = flatten(w)

func = theano.function([],y)

dimensions = 1
flattened = flatten (w,dimensions)
print (flattened)
print (flattened.shape)


d = T.constant(2)
y = flatten(x,t)

func = theano.function([w,d],y)

dimensions = 1
new = numpy.random.RandomState(1).rand(3, 2, 5, 5) # Max pool will take the last two indexes for the pooling

flattened = func (new,dimensions)
print (flattened)
print (flattened.shape)
