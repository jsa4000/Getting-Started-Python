import theano
import theano.tensor as T
import numpy as np

# The thing is that I have to do padding for all the features in the images and outputs
invals = numpy.random.RandomState(1).rand(3, 2, 5, 5) # Max pool will take the last two indexes for the pooling

##############
### NUMPY ###
##############

padding = 2
matrix = np.random.RandomState(1).rand(5, 5) 
zeromatrix = np.zeros((5 + padding , 5 + padding)) 

print (matrix)
print (zeromatrix)

offset = padding/2
print (zeromatrix[offset : -offset, offset : -offset])

zeromatrix[offset : -offset, offset : -offset] = matrix
print (zeromatrix)

##############
### THEANO ###
##############

def padding (x, padding):
    #Create a zero matrix with the padding
    y = T.zeros((x.shape[0] + padding , x.shape[1] + padding))
    # Get the offset to apply to the matrix
    offset = padding // 2 # integer division
    #y[offset : -offset, offset : -offset] = x
    return T.set_subtensor(y[offset : -offset, offset : -offset], x)

x = T.matrix("X")
y = T.matrix("Y")
p = T.iscalar("p")

y = padding(x,p)

func = theano.function([x,p],y)

padding = 2
matrix = np.random.RandomState(1).rand(5, 5) 

print (func(matrix,padding))

#################################
# Now the same but ufing the last two values of the shape ofr 4D Matrix

paddin = 2
offset = 1 # 2/1
invals = numpy.random.RandomState(1).rand(3, 2, 5, 5) # Max pool will take the last two indexes for the pooling

# Two last item of the shape must sum 1
#zeromatrix = np.zeros((5 + padding , 5 + padding)) 
new_shape = []
for i, item in enumerate(invals.shape):
    if (i + 2 >= len(invals.shape)):  # Two last dimensions
        new_shape.append(item + (offset* 2) )
    else:
        new_shape.append(item)

print (tuple(new_shape))

zeromatrix = np.zeros(tuple(new_shape)) 

print (zeromatrix[:,:,offset : -offset, offset : -offset])
print (invals)

zeromatrix[:,:, offset : -offset, offset : -offset] = invals
print (zeromatrix)

##

paddin = 2
offset = 1 # 2/1
invals = numpy.random.RandomState(1).rand(3, 2, 5, 5) # Max pool will take the last two indexes for the pooling

new_shape = tuple( item + (offset* 2) if (index + 2 >= len(invals.shape)) else item for index,item in enumerate(invals.shape) )
print (new_shape)

new_slice = tuple(slice(offset,-offset)  if (index + 2 >= len(new_shape)) else slice(None) for index,item in enumerate(new_shape) )
print (new_slice)

zeromatrix = np.zeros(new_shape) 
print (zeromatrix)

print (zeromatrix[:,:,offset : -offset, offset : -offset])
print (invals)

print (zeromatrix[:,:, offset : -offset, offset : -offset])
print (zeromatrix[new_slice])


zeromatrix[new_slice] = invals
print (zeromatrix)


def padNumpy(x,offset):
    new_shape = tuple( item + (offset* 2) if (index + 2 >= len(x.shape)) else item for index,item in enumerate(x.shape) )
    new_slice = tuple(slice(offset,-offset)  if (index + 2 >= len(new_shape)) else slice(None) for index,item in enumerate(new_shape) )
    zeromatrix = np.zeros(new_shape) 
    zeromatrix[new_slice] = x
    return zeromatrix


offset = 1
invals = numpy.random.RandomState(1).rand(3, 2, 5, 5) # Max pool will take the last two indexes for the pooling
print(padNumpy(invals,offset))

matrix = np.random.RandomState(1).rand(5, 5) 
print(padNumpy(matrix,offset))

## THEANO ##

def padding (x, offset):
    length = x.shape[0]
    new_shape = tuple( item + (offset* 2) if (T.ge(index + 2 ,length)) else item for index,item in enumerate(x.shape) )
    new_slice = tuple(slice(offset,-offset)  if (index + 2 >= len(new_shape)) else slice(None) for index,item in enumerate(new_shape) )
    y = T.zeros(new_shape)
    return T.set_subtensor(y[new_slice], x)

x = T.matrix("X")
y = T.matrix("Y")
p = T.iscalar("p")

y = padding(x,p)

func = theano.function([x,p],y)

offset = 1

matrix = np.random.RandomState(1).rand(5, 5) 
print(func(matrix,offset))



def padding (x, x_len, offset):
    new_shape = tuple( item + (offset* 2) if (T.ge(index + 2 ,x_len)) else item for index,item in enumerate(x.shape) )
    new_slice = tuple(slice(offset,-offset)  if (index + 2 >= len(new_shape)) else slice(None) for index,item in enumerate(new_shape) )
    y = T.zeros(new_shape)
    return T.set_subtensor(y[new_slice], x)

x = T.tensor4("X")
y = T.tensor4("Y")
l = T.iscalar("L")
p = T.iscalar("P")

y = padding(x,l,p)

func = theano.function([x,l,p],y)

invals = numpy.random.RandomState(1).rand(3, 2, 5, 5) # Max pool will take the last two indexes for the pooling
print(func(invals,len(invals.shape),offset))


def padding (x, offset):
    #new_shape = tuple( item + (offset* 2) if (T.ge(index + 2 ,x.shape[0])) else item for index,item in enumerate(x.shape) )
    #new_shape = tuple( item + (offset* 2) if (index + 2 >= 4) else item for index,item in enumerate(x.shape) )
    new_shape = tuple( item + (offset* 2) if (index + 2 >= x.ndim) else item for index,item in enumerate(x.shape) )
    new_slice = tuple(slice(offset,-offset)  if (index + 2 >= len(new_shape)) else slice(None) for index,item in enumerate(new_shape) )
    y = T.zeros(new_shape)
    return T.set_subtensor(y[new_slice], x)

x = T.tensor4("X")
y = T.tensor4("Y")
p = T.iscalar("P")

y = padding(x,p)

func = theano.function([x,p],y)

invals = numpy.random.RandomState(1).rand(3, 2, 5, 5) # Max pool will take the last two indexes for the pooling
print(func(invals,offset))



############### FINAL
framework = 0
def padding (x, offset = 1):
    if (framework == 0):
        new_shape = tuple( item + (offset* 2) if (index + 2 >= x.ndim) else item for index,item in enumerate(x.shape))
        new_slice = tuple(slice(offset,-offset)  if (index + 2 >= len(new_shape)) else slice(None) for index,item in enumerate(new_shape) )
        zeromatrix = np.zeros(new_shape) 
        zeromatrix[new_slice] = x
        return zeromatrix
    elif (framework == 1):
        new_shape = tuple( item + (offset* 2) if (index + 2 >= x.ndim) else item for index,item in enumerate(x.shape) )
        new_slice = tuple(slice(offset,-offset)  if (index + 2 >= len(new_shape)) else slice(None) for index,item in enumerate(new_shape) )
        y = T.zeros(new_shape)
        return T.set_subtensor(y[new_slice], x)
    else:
        raise DeepPyException("Framework doesn't found") 

#x = T.tensor4("X")
#y = T.tensor4("Y")
#p = T.iscalar("P")

#y = padding(x,p)

#func = theano.function([x,p],y)

#invals = numpy.random.RandomState(1).rand(3, 2, 5, 5) # Max pool will take the last two indexes for the pooling
#print(func(invals,offset))

#x = T.matrix("X")
#y = T.matrix("Y")
#p = T.iscalar("p")

#y = padding(x,p)

#func = theano.function([x,p],y)

#offset = 1

#matrix = np.random.RandomState(1).rand(5, 5) 
#print(func(matrix,offset))

offset = 1
invals = numpy.random.RandomState(1).rand(3, 2, 5, 5) # Max pool will take the last two indexes for the pooling
print(padding(invals,offset))

matrix = np.random.RandomState(1).rand(5, 5) 
print(padding(matrix,offset))