import theano
import theano.tensor as T
import numpy as np

# It's more powerful than
# dimshuffle = transpose + reshape

#dimshuffle('x', 0, 'x', 'x')) 


matrix = np.asarray([[1,2,3],[4,5,6],[7,8,9]])

#Doing the stuf with transpose
matrixT = matrix.transpose((1,0))
matrixT2 = matrix.T

# These two columns are the same

print(matrix)
print(matrixT)
print(matrixT2)

# Doing the same stuff with reshape
# None = newaxis


vector = np.asarray([0,1,2,3,4])
vectorR = vector[:,None,None] # Doing indexing and broadcasting
vectorR2 = vector.reshape((vector.shape[0],1,1))

print (vector)
print (vectorR)
print (vectorR2)

vectorR3 = vector.reshape((1, vector.shape[0],1,1))
print (vectorR3)

vectorR4 = vector[None, :,None,None] # Doing indexing and broadcasting
print (vectorR4)

# To implemnt diffsuffle I can use the same nomenclature as used in numpy, usong none, etc..



#Matrix to be transformed
matrix = np.asarray([[[1,2,3,4],[5,6,7,8],[9,10,11,12]]])
print (matrix.shape)

# (1L, 3L, 4L)
# This is what I really expect
# new_shape => (1,4,1,3) # Where the third value is new dim to add

# This is what I whould punt into dimshuffle from theano framework 
#  -> dimshuffle(0, 2, 'x', 1))  -> 


# THis is what I have to do in my function -> 'x' -> None
ds = (0,2, None, 1) # Similar to thano
print (ds)
# Transpose using numpy

# Remove all the None values from the original to do the transpose
t_shape = tuple(dim for dim in ds if dim is not None)
print (t_shape)

# Transpose the matrix with the specified axis values
matrixT = matrix.transpose(t_shape)
print (matrixT)
print (matrixT.shape)

# Finally Broadcasting using numpy
new_shape = tuple (-1 if axis is None else matrixT.shape[axis] for index, axis in enumerate(ds))
matrixR = matrixT.reshape(new_shape)

print (matrixR)
print (matrixR.shape)



def dimshuffle (x, shape):
    """
        This function will allow the possibility to dimsuffle any matrix to the given shape (if it's possible)
        This function act like dimsuffle in Theano. Instead 'X' (theano) or -1 (numpy) a None value will be passed by value if new dimension.
        Parameters:
            x: matrix
                Matrix input that will be reshaped and transposed
            shape: tuple
                Shape for the desired new shape and transpose. None or np.newaxis will be used to add a new dimension to an axis.
    """
    shapeT = tuple(dim for dim in shape if dim is not None)
    xT = x.transpose(shapeT)
    new_shape = tuple (-1 if axis is None else x.shape[axis] for index, axis in enumerate(shape))
    return xT.reshape(new_shape)


# New shape wit the axis transposed
ds = (0,2, None, 1) # Similar to thano
print (ds)

# The mtrix which the dissshuffle will be applied
matrix = np.asarray([[[1,2,3,4],[5,6,7,8],[9,10,11,12]]])
print (matrix.shape)
# (1L, 3L, 4L)
# axis[0] = 1, axis[1] = 3, axis[2] = 4

# So the output matrix will have the following shape
# (1, 4, 1, 3)  => Replace the axis with the size of that axis in matrix.shape

matrixD = dimshuffle(matrix, ds)
print (matrixD)
print (matrixD.shape)




ds = (0,2, None, 1) # Similar to thano
nds = np.asarray(ds)
nds[nds == None] = 'X'
print (tuple(nds))

x = matrix
shape = ds
new_shape = tuple ('x' if axis is None else x.shape[axis] for index, axis in enumerate(shape))
print (new_shape)



ds = (0,2, None, 1) # Similar to thano
def dimshuffle (x):
    """
        This function will allow the possibility to dimsuffle any matrix to the given shape (if it's possible)
        This function act like dimsuffle in Theano. Instead 'X' (theano) or -1 (numpy) a None value will be passed by value if new dimension.
        Parameters:
            x: matrix
                Matrix input that will be reshaped and transposed
            shape: tuple
                Shape for the desired new shape and transpose. None or np.newaxis will be used to add a new dimension to an axis.
    """
    new_shape = tuple ('x' if axis is None else axis for axis in ds)
    return x.dimshuffle(*new_shape)

x = T.tensor3()
y = T.tensor4()

y = dimshuffle(x)

func = theano.function([x],y)

matrix = np.asarray([[[1,2,3,4],[5,6,7,8],[9,10,11,12]]])

matrixD = func(matrix)

print (matrixD)
print (matrixD.shape)


def dimshuffle(x, shape):
    """
        This function will allow the possibility to dimsuffle any matrix to the given shape (if it's possible)
        This function act like dimsuffle in Theano. Instead 'X' (theano) or -1 (numpy) a None value will be passed by value if new dimension.
        Parameters:
            x: matrix
                Matrix input that will be reshaped and transposed
            shape: tuple
                Shape for the desired new shape and transpose. None or np.newaxis will be used to add a new dimension to an axis.
    """
    if (framework == 0):
        #Get the new shape once the layer has been flattened
        shapeT = tuple(dim for dim in shape if dim is not None)
        xT = x.transpose(shapeT)
        new_shape = tuple (-1 if axis is None else x.shape[axis] for index, axis in enumerate(shape))
        return xT.reshape(new_shape)
    elif (framework == 1):
        new_shape = tuple ('x' if axis is None else axis for axis in ds)
        return x.dimshuffle(*new_shape)
    else:
        raise DeepPyException("Framework doesn't found") 







