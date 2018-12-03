import tensorflow as tf
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '4'

# Define a single queue with two components to store the input data.
q_data = tf.FIFOQueue(100000, [tf.float32, tf.float32])

# We will use these placeholders to enqueue input data.
placeholder_x = tf.placeholder(tf.float32, shape=[None])
placeholder_y = tf.placeholder(tf.float32, shape=[None])
enqueue_data_op = q_data.enqueue_many([placeholder_x, placeholder_y])

gs = tf.Variable(0)
w = tf.Variable(0.)
b = tf.Variable(0.)
optimizer = tf.train.GradientDescentOptimizer(0.05)

# Construct the while loop.
def cond(i):
    return i < 10

def body(i):
    # Dequeue a single new example each iteration.
    x, y = q_data.dequeue()
    # Compute the loss and gradient update based on the current example.
    loss = (tf.add(tf.multiply(x, w), b) - y) ** 2
    train_op = optimizer.minimize(loss, global_step=gs)
    # Ensure that the update is applied before continuing.
    with tf.control_dependencies([train_op]):
        return i + 1

loop = tf.while_loop(cond, body, [tf.constant(0)])

data = [k * 1. for k in range(10)]

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(1):
        # NOTE: Constructing the enqueue op ahead of time avoids adding
        # (potentially many) copies of `data` to the graph.
        sess.run(enqueue_data_op,
                 feed_dict={placeholder_x: data, placeholder_y: data})
    print (sess.run([gs, w, b]))  # Prints before-loop values.
    sess.run(loop)
    print (sess.run([gs, w, b]))  # Prints after-loop values.
