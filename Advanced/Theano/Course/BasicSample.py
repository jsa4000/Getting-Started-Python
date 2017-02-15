import theano
from theano import tensor as T

# THEANO: NUMERICAL & SYMBOLIC COMPUTATION 
# NUMPY: NUMERICAL COMPUTATION

# First declare the symbolic computation
a = T.scalar()
b = T.scalar()
y = a * b

# Create the function in theano
multiply = theano.function([a,b],y)

# Second perform numerical computation
print (multiply(2,4))
print (multiply(5,4))
