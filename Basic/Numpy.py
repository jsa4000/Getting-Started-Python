"""
In this class I will be using Numpy

NumPy is the fundamental package for scientific computing with Python.
It contains among other things:

- A powerful N-dimensional array object. Also called Tensor.
- Sophisticated (broadcasting) functions and vectorization.
- Tools for integrating C/C++ and Fortran code
- useful linear algebra, Fourier transform, and random number capabilities

Besides its obvious scientific uses, NumPy can also be used as an efficient
multi-dimensional container of generic data. Arbitrary data-types can be defined.
This allows NumPy to seamlessly and speedily integrate with a wide variety of databases.
"""
# pylint: disable=invalid-name
import numpy as np

# If you're getting errors with pylint try the following
# pip install [package] --upgrade
# pip install pylint --upgrade

# The type of a nump array is ndarray.
# In order to create an array you can convert from python sequnces.
# NOTE: Numpy are numerical arrays N-dimensions, so it doesn't support stirngs
#       or other types nos strcutures.

py_array_f = [[1.2, 2], [3.5, 4], [5.2, 6.6]]
print(type(py_array_f))
print(py_array_f)
py_nparray_f = np.asarray(py_array_f)

# Convert to Numpy Array. Numpy will detect the type automatically
# If all values have not dot quotation, ej. 0.4. The returned matrix will
# be integer, otherwise the type resturned will be float (32 or 64 will depend
# on the floating point specification on your CPU, OS, GPU, etc..)

#You can transform the sequences by using array or asarray
#    The definition of asarray is:
#           def asarray(a, dtype=None, order=None):
#                return array(a, dtype, copy=False, order=order)
#   So it is like array, except it has fewer options, and copy = False. array has copy = True by default.
#   Main difference is that array (by default) will make a copy of the object, while asarray will not unless necessary.


# IN GENERAL
# array -> This returns the same array but wil modfify some attrbiutes.
#           Any mofification doen in this ndarray will modify the original
# asarray -> Create a clone or a copy. This won't modigy the original


#To test it we can see if the original list it's modified using both methods
py_array = [[2, 2], [3, 4], [5, 6]]
print(py_array)
#py_array[0][1] = 56
print(py_array)

py_asarray = np.asarray(py_array)

# Instance of py_asarray. Ig this is modified then py_asarray is modified too.
py_a_array = np.array(py_asarray)

# Copy or clone from py_asarray
py_asarray_2 = np.asarray(py_asarray)

#py_asarray[0][1] = 56

# Following sentance modify py_asarray since it's an instance
#py_a_array[0][1] = 56

# Following sentance won't modify the original py_asarray since it's copy
py_asarray_2[0][1] = 56

# Original has been modified? -> 
print(py_asarray)
#THe original from pytthon list won't be modified
print(py_array)


py_array = [[2, 2], [3, 4], [5, 6]]
np_array = np.asarray(py_array)
print(type(np_array))
print(np_array)

#To get the shape of the matrix. (Tuple is returned with the spahe)
print(np_array.shape)
# To get the type of the current Matrix
print(np_array.dtype)
# To get the dimensions of the current matrix
print(np_array.ndim)
# To get the total imtes of the current matrix
print(np_array.size)

# Multi-arrays or Tensor are useful for image processing and vectorization
# Operations are must faster

py_array_3x3x3 = [[[2], [3]], [[6], [0]], \
                 [[3], [2]], [[0], [4]], \
                 [[3], [7]], [[9], [0]]]
py_ndarray_3x3x3 = np.asarray(py_array_3x3x3)

print(py_ndarray_3x3x3.shape) # (6,2,1)
print(py_ndarray_3x3x3.ndim) # 3 dimensions
print(py_ndarray_3x3x3.size) # 12 items 6 * 2 *  1


#You can also specify the type to cast the inputs of the NMatrix
np_array = np.asarray(py_nparray_f, dtype=np.int32)

# Get the original float ndarray
print(py_nparray_f.dtype)
print(py_nparray_f)

# Get the casted to int 32
print(np_array.dtype)
print(np_array)

# Creata a new copy but Numpy Matrix
np_copy_array = np_array.copy()

# You can also create new numpy arrays. 
print (np.zeros((2,2)))
print (np.ones((2,2)))

#We can also rehape the array or the Marix
loc, sigma, size = 0, 0.2, 12
normal_dist = np.random.normal(scale=sigma,size=size)
print(normal_dist)
normal_matrix = np.reshape(normal_dist, (6, 2, 1))
print(normal_matrix)
# Or Flatter again the  Matrix
normal_dist = normal_matrix.flatten()
# Or Flatter again the  Matrix (revel seems more quickest)
normal_dist_ravel = normal_matrix.ravel()

print(normal_dist)
print(normal_dist_ravel)
#Min Max value
print(normal_matrix.max())
print(normal_matrix.min())

# You can do basic operations using Numpy such as Dot product
six_matrix = np.ones((3,3)) * 6
print(six_matrix)

# Create a linear distribution
linear_dist = np.linspace(0,12, 6) # Step = 2
linear_dist_1 = np.linspace(0,12, 13) # Step = 1
linear_dist_2 = np.linspace(0,12, 24) # Step = 0.5
print(linear_dist)
print(linear_dist_1)
print(linear_dist_2)


# Create a log distribution
log_dist = np.logspace(0,12, 6) # Step = 2
log_dist_1 = np.logspace(0,12, 13) # Step = 1
log_dist_2 = np.logspace(0,12, 24) # Step = 0.5
print(log_dist)
print(log_dist_1)
print(log_dist_2)

#Create a range value of values. Similar to linear distribution
#This will allow to go through the range defined with the step defined.abs
# This won't fit the rante between the values defines like linead distribution
dist_range = np.arange(30)
print(dist_range)
dist_range = np.arange(0,12,2)
print(dist_range)

# Let's test some values for fit values

#Clip it's similar to clamp. This will limit your values to tghe
#    numpy.clip(a, a_min, a_max, out=None)[source]
fit_dist = np.clip(linear_dist,0,3)
print(fit_dist)

# umpy.interp(x, xp, fp, left=None, right=None, period=None)
#fit_dist = np.interp(linear_dist,0,3)
#print(fit_dist)

# You can get the min max values for the axis indepently
#Those values are the mean of all of them
py_array = [[2, 2], [3, 4], [5, 6]]
mins = np.min(py_array, axis=0)
maxs = np.max(py_array, axis=0)
print(mins)
print(maxs)

#You can also compute the meain. Also you can use the axis you want to use.
mean = np.mean(py_ndarray_3x3x3)
print(mean)

# Operations
a = np.reshape(np.arange(4), (2,2))
print(a)
b = np.reshape(np.arange(start=4,stop=8), (2,2))
print(b)

print("Let's operate with the Matrixes")
# dot product (shape it's important)
dot_mat = np.dot(a, b)
print(dot_mat)
# multiplication
dot_mult = np.matmul(a, b)
print(dot_mult)

#matmul differs from dot in two important ways.
#    Multiplication by scalars is not allowed.
#    Stacks of matrices are broadcast together as if the matrices were elements.

# normal multi element by element. - i1 * j1, i2 * j2
dot_mult_v2 = a * b
print(dot_mult_v2)
#Scalar multiplication
dot_mult_v3 = a * 3
print(dot_mult_v3)
#Transpose
a_1 = np.arange(4)
print(a_1.shape)
print(a_1)
print(a_1.T)
print(a_1.T.shape)

#Identity Matrix NxN
print(np.identity(3))

#There is an interesting article about performances in Numpy
#   http://ipython-books.github.io/featured-01/

# Getting the Best Performance out of NumPy 


# n-place and implicit copy operations


# Broadcasting rules

# Broadcasting rules allow you to make computations on arrays with different
#  but compatible shapes. In other words, you don't always need to reshape or
#  tile your arrays to make their shapes match. The following example illustrates
#  two ways of doing an outer product between two vectors: the first method
#  involves array tiling, the second one involves broadcasting. The last method
#  is significantly faster.

n = 10
a = np.arange(n)
print (a)
# This will add a column to the array with the values 
ac = a[:, np.newaxis]
print (ac)
# This will add a new row with a column with the distribution
ar = a[np.newaxis, :]
print (ar)


#Making efficient selections in arrays with NumPy (fancy indexing)

# NumPy arrays can be indexed with slices, but also with boolean or
#  integer arrays (masks). This method is called fancy indexing.
#
# FANCY INDEXING CREATE COPIES NOT VIEWS.

n, d = 100000, 100
a = np.random.random_sample((n, d)); aid = id(a)
#Get the selection directly using the indexing
b1 = a[::10]
#Crate a new matrix for filtering the items
b2 = a[np.arange(0, n, 10)]
# Compare both matrix
np.array_equal(b1, b2)

#Boolean Maks
op_array = np.arange(5)
print(op_array)
#Create a mask
mask = op_array > 2
# Apply the mask
fanc_mask = op_array[mask]
# This mask will return the elements > 2
print(fanc_mask)

# It can return the values from an array of integers
elements = [1, 1, 4, 4]
fanc_mask = op_array[elements]
print(fanc_mask)


#Copies and views

# A slicing operation creates a view on the original array, which is just
#  a way of accessing array data. Thus the original array is not copied
#  in memory. You can use np.may_share_memory() to check if two arrays share
#  the same memory block. Note however, that this uses heuristics and may
#  give you false positives.

# View: allows to make changes to the original matric or array
# Copy: it's just a copy,so the original won't be affected.

a = np.arange(10)
b = a[::2]

#Original
print(a)
# View
print(b)

#Modify an element of the current view
b[0] = 12

# Print again the values. the original adn the view has been changed
#Original
print(a)
# View
print(b)

# It's important to know the axis. 
# Almost every opeartion allow to specify the axis for the current op

#To delete a row or column
orig_m = np.arange(20).reshape(5,4)
print (orig_m)

# Delete the firsrt column (axis = 1)
del_m = np.delete(orig_m,0,axis=1)
print (del_m)

# Delete the firsrt row (axis = 0)
del_m = np.delete(orig_m,0,axis=0)
print (del_m)

#Squeze will remove one dimension
py_array_3x3x3 = [[[2], [3]], [[6], [0]], \
                 [[3], [2]], [[0], [4]], \
                 [[3], [7]], [[9], [0]]]
py_ndarray_3x3x3 = np.asarray(py_array_3x3x3)
print(py_ndarray_3x3x3.shape)
print(py_ndarray_3x3x3.ndim)
#print(py_array_3x3x3)

squeezed = np.squeeze(py_array_3x3x3)
print(squeezed.shape)
print(squeezed.ndim)
#print(squeezed)

#Matrices can also be concatenated using the following method
