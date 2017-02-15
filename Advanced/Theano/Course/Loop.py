import theano
import theano.tensor as T

# http://nbviewer.jupyter.org/gist/triangleinequality/1350873eebea33973e41
# http://deeplearning.net/software/theano/library/scan.html


#####################
## EXAMPLE 01
######################

# This loop is what i need to do in theano
#result = 1
#for i in range(k):
#    result = result * A

k = T.iscalar("k")
A = T.vector("A")

def power(x, a):
    return x * a
    

# Symbolic description of the result
#result, updates = theano.scan(fn=lambda prior_result, A: prior_result * A,
#                              outputs_info=T.ones_like(A),
#                              non_sequences=A,
#                              n_steps=k)

result, updates = theano.scan(fn=power,
                              outputs_info=T.ones_like(A),
                              non_sequences=A,
                              n_steps=k)

#Let us go through the example line by line. What we did is first to construct a function (using a lambda expression) 
#that given prior_result and A returns prior_result * A. The order of parameters is fixed by scan: the output of the prior
# call to fn (or the initial value, initially) is the first parameter, followed by all non-sequences.

#Next we initialize the output as a tensor with same shape and dtype as A, filled with ones. We give A to scan as a non sequence 
#parameter and specify the number of steps k to iterate over our lambda expression.

#Scan returns a tuple containing our result (result) and a dictionary of updates (empty in this case). Note that the result is 
#not a matrix, but a 3D tensor containing the value of A**k for each step. We want the last value (after k steps) so we compile 
#a function to return just that. Note that there is an optimization, that at compile time will detect that you are using just the 
#last value of the result and ensure that scan does not store all the intermediate values that are used. So do not worry if A and k are large.


# We only care about A**k, but scan has provided us with A**1 through A**k.
# Discard the values that we don't care about. Scan is smart enough to
# notice this and not waste memory saving them.
final_result = result[-1]

# compiled function that returns A**k
fpower = theano.function(inputs=[A,k], outputs=final_result, updates=updates)

print(fpower(range(10),2))
print(fpower(range(10),4))


#####################
## EXAMPLE 02
######################

import numpy

coefficients = theano.tensor.vector("coefficients")
x = T.scalar("x")

max_coefficients_supported = 10000

# Generate the components of the polynomial
components, updates = theano.scan(fn=lambda coefficient, power, free_variable: coefficient * (free_variable ** power),
                                  outputs_info=None,
                                  sequences=[coefficients, theano.tensor.arange(max_coefficients_supported)],
                                  non_sequences=x)
# Sum them up
polynomial = components.sum()

# Compile a function
calculate_polynomial = theano.function(inputs=[coefficients, x], outputs=polynomial)

# Test
test_coefficients = numpy.asarray([1, 0, 2], dtype=numpy.float32)
test_value = 3
print(calculate_polynomial(test_coefficients, test_value))
print(1.0 * (3 ** 0) + 0.0 * (3 ** 1) + 2.0 * (3 ** 2))


# Second, there is no accumulation of results, we can set outputs_info to None. This indicates to scan that it 
# doesn’t need to pass the prior result to fn.

# The general order of function parameters to fn is:

#     sequences (if any), prior result(s) (if needed), non-sequences (if any)

# Fourth, given multiple sequences of uneven lengths, scan will truncate to the shortest of them. This makes it safe
# to pass a very long arange, which we need to do for generality, since arange must have its length specified at creation time.


