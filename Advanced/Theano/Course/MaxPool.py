import theano
import theano.tensor as T
from theano.tensor.signal import downsample
import numpy 

# Now what the Max Pooling algorithm does in a matrix.
input = T.dtensor4('input')
maxpool_shape = (2, 2)
pool_out = downsample.max_pool_2d(input, maxpool_shape, ignore_border=True)
f = theano.function([input],pool_out)

invals = numpy.random.RandomState(1).rand(3, 2, 5, 5) # Max pool will take the last two indexes for the pooling

#print invals

# Maxpool take a Matrix (multiple featutes, convolutions, etc), let say 10x10. 
    # An for all existing matrix there are in the 4D matrix [:,:;m;n]
    # Generate a new matrix (with same shape) downsmapled with the maxpool_shape defined.
    # For that it will divide the matrix 10x10 into the maxpool defined and will teake the maximun value.

print 'With ignore_border set to True:'
print 'invals[0, 0, :, :] =\n', invals[0, 0, :, :]
print 'output[0, 0, :, :] =\n', f(invals)[0, 0, :, :]

pool_out = downsample.max_pool_2d(input, maxpool_shape, ignore_border=False)
f = theano.function([input],pool_out)
print 'With ignore_border set to False:'
print 'invals[1, 0, :, :] =\n ', invals[1, 0, :, :]
print 'output[1, 0, :, :] =\n ', f(invals)[1, 0, :, :]

# Important note:
#   - If matrix is 31 x 31 the the max pool result with (2,2) of max pool will generate a new
#     matrix with 15x15, so it's like only get the integer part dividing Int(31/2) = 15
#   -  If matrix is 31 x 31 the the max pool result with (3,3) of max pool will generate a new
#     matrix with 10x10 -> Unt(31/3) = 10

