import tensorflow as tf
from math import sqrt

# For reference use the TensorFlow documentation at
# - https://www.tensorflow.org/api_docs/python/tf

# Function to comute 
# ( bias * param1 ) * sqrt( weight / param2)

params = {'param1':0.2, 'param2':4.0, 'bias':2.0, 'weight':8.0}
print(params)

# Input Parameters (scalars)
bias = tf.placeholder(tf.float32, name='bias')
weight = tf.placeholder(tf.float32, name='weight')

# Constant Values (hyper-parameters)
param1 = tf.constant(params['param1'], name='param1')
param2 = tf.constant(params['param2'], name='param2')

# Computation graph
#   div and multiply operation allows to set scalars, vector or matrices using tf optimizations
with tf.name_scope("function"):
    with tf.name_scope("left_side"):
        left = tf.multiply(bias, param1, name='multiply') 
    with tf.name_scope("right_side"):
        right = tf.sqrt(tf.div(weight, param2, name='divide'),name='sqrt')
    function =  tf.multiply(left, right, name='multiply')

# Create the tensor flow session
with tf.Session() as session:
    # Create the FileWriter for the graph
    writer = tf.summary.FileWriter(".outputs/output2", session.graph)

    # Build the symbolic model (pipeline), execute the graph, and print the result
    tfOutput = session.run(function, {bias:params['bias'], weight:params['weight']})
    print('From TensorFlow = {}'.format(tfOutput))

    expectedOutput = ( params['bias'] * params['param1'] ) * sqrt( params['weight'] / params['param2'])
    print('From Expected = {}'.format(expectedOutput))
    # Close writes at the end
    writer.close()

#Use tensorboard --logdir=.outputs/output2