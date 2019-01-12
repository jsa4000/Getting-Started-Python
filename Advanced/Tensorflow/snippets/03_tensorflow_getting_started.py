import tensorflow as tf

# It must be provided the shape of each type of input as the second argument.
#     `a` to be a 2-dimensional matrix, with 2 rows and 1 column
#     `b` to be a matrix with 1 row and 2 columns
#     `c` to be a float32 scalar
a = tf.placeholder(tf.float32, shape=(2,1))
b = tf.placeholder(tf.float32, shape=(1,2))
c = tf.placeholder(tf.float32)

# Define the outputs `d` and `e` as the matrix multiplication operations,
# with the inputs coming from `a`, `b` and `c`
d = tf.matmul(a, b)
e = c * d

# Create the tensor flow session
session = tf.Session()

# Build the symbolic model (pipeline), execute the graph, and print the result
print(session.run(e, {a: [[1],[2]], b:[[3,4]], c:2}))