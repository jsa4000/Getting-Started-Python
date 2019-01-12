import tensorflow as tf
import numpy as np

def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)

# x and y are placeholders for our training data
x = tf.placeholder("float")
y = tf.placeholder("float")
# w is the variable storing our values. It is initialized with starting "guesses"
# w[0] is the "a" in our equation, w[1] is the "b"
w = tf.Variable([1.0, 2.0], name="w")
variable_summaries(w)
# Our model of y = a*x + b
with tf.name_scope('model'):
    y_model = tf.multiply(x, w[0]) + w[1]
    tf.summary.histogram('pre_activations', y_model)

# Our error is defined as the square of the differences
with tf.name_scope('error'):
    error = tf.square(y - y_model)
tf.summary.scalar('error', error)

# The Gradient Descent Optimizer does the heavy lifting
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(error)

# Normal TensorFlow - initialize values, create a session and run the model
model = tf.global_variables_initializer()

with tf.Session() as session:
    merged_summary = tf.summary.merge_all()

     # Create the FileWriter for the graph
    writer = tf.summary.FileWriter(".outputs/output3", session.graph)

    session.run(model)
    for i in range(1000):
        x_value = np.random.rand()
        y_value = x_value * 2 + 6
        _, summary  = session.run([train_op, merged_summary], feed_dict={x: x_value, y: y_value})
        # Add output for merged variables into the summary (tensorboard)
        writer.add_summary(summary, i)

    w_value = session.run(w)
    print("Predicted model: {a:.3f}x + {b:.3f}".format(a=w_value[0], b=w_value[1]))

    # Close writes at the end
    writer.close()