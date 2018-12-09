import os
import time
import math
from datetime import datetime
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# os.environ['CUDA_VISIBLE_DEVICES'] = '2'

PID = os.getpid()
print('Program pid:', PID)
print('Pause here to enter DBG')
#os.system("read")

CONFIG = tf.ConfigProto()
CONFIG.gpu_options.allow_growth=True
# CONFIG.log_device_placement=True

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch_size', 100,
                            """Batch size.""")
tf.app.flags.DEFINE_integer('num_batches', 50,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('print_steps', 1,
                            """Number of steps when log.""")

def simple_dnn():
    # Import data
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    # Create the model
    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10])

    # Build the graph
    with tf.name_scope('fc'):
        W = tf.Variable(tf.zeros([784, 10]))
        b = tf.Variable(tf.zeros([10]))

        y = tf.matmul(x, W) + b

    with tf.name_scope('loss'):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y))

    with tf.name_scope('GD_optimizer'):
        train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    
    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
    
    accuracy = tf.reduce_mean(correct_prediction)

    # name = get_names()
    # for item in name:
    #     print(item)

    # tensor = get_tensors()
    # for item in tensor:
    #     print(item)
    
    with tf.Session(config=CONFIG) as sess:
        num_steps_burn_in = 10
        total_duration = 0.0
        total_duration_squared = 0.0
        sess.run(tf.global_variables_initializer())
        for i in range(FLAGS.num_batches + num_steps_burn_in):
            batch = mnist.train.next_batch(FLAGS.batch_size)
            start_time = time.time()
            train_step.run(feed_dict={x: batch[0], y_: batch[1]})
            duration = time.time() - start_time

            if i > num_steps_burn_in:
                if i % FLAGS.print_steps == 0:
                    train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1]})
                    print('%s: step %d, duration = %.3f, training accuracy %g' % (datetime.now(), i - num_steps_burn_in, duration, train_accuracy))
                total_duration += duration
                total_duration_squared += duration * duration

        mn = total_duration / FLAGS.num_batches
        vr = total_duration_squared / FLAGS.num_batches - mn * mn
        sd = math.sqrt(vr)
        print('%s: %s across %d steps, %.3f +/- %.3f sec / batch' % (datetime.now(), "Forward_backword", FLAGS.num_batches, mn, sd))
        print('Simple DNN test accuracy %g' % accuracy.eval(feed_dict={
            x: mnist.test.images, y_: mnist.test.labels}))

def simple_cnn():
    # Import data
    mnist = input_data.read_data_sets('MNIST_data')

    # Create the model
    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.int64, [None])

    # Build the graph for the deep net
    #y_conv, keep_prob = deepnn(x)

    # Reshape to use within a convolutional neural net.
    # Last dimension is for "features" - there is only one here, since images are
    # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
    with tf.name_scope('reshape'):
        x_image = tf.reshape(x, [-1, 28, 28, 1])

    # First convolutional layer - maps one grayscale image to 32 feature maps.
    with tf.name_scope('conv1'):
        W_conv1 = weight_variable([5, 5, 1, 32])
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

    # Pooling layer - downsamples by 2X.
    with tf.name_scope('pool1'):
        h_pool1 = max_pool_2x2(h_conv1)

    # Second convolutional layer -- maps 32 feature maps to 64.
    with tf.name_scope('conv2'):
        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    # Second pooling layer.
    with tf.name_scope('pool2'):
        h_pool2 = max_pool_2x2(h_conv2)

    # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
    # is down to 7x7x64 feature maps -- maps this to 1024 features.
    with tf.name_scope('fc1'):
        W_fc1 = weight_variable([7 * 7 * 64, 1024])
        b_fc1 = bias_variable([1024])

        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Dropout - controls the complexity of the model, prevents co-adaptation of features.
    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    
    # Map the 1024 features to 10 classes, one for each digit
    with tf.name_scope('fc2'):
        W_fc2 = weight_variable([1024, 10])
        b_fc2 = bias_variable([10])
    
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    with tf.name_scope('loss'):
        cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(
            labels=y_, logits=y_conv))

    with tf.name_scope('adam_optimizer'):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), y_)
        correct_prediction = tf.cast(correct_prediction, tf.float32)

    accuracy = tf.reduce_mean(correct_prediction)

    # graph_location = tempfile.mkdtemp()
    # print('Saving graph to: %s' % graph_location)
    # train_writer = tf.summary.FileWriter(graph_location)
    # train_writer.add_graph(tf.get_default_graph())

    
    with tf.Session(config=CONFIG) as sess:
        num_steps_burn_in = 10
        total_duration = 0.0
        total_duration_squared = 0.0
        sess.run(tf.global_variables_initializer())
        for i in range(FLAGS.num_batches + num_steps_burn_in):
            batch = mnist.train.next_batch(FLAGS.batch_size)
            start_time = time.time()
            train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
            duration = time.time() - start_time

            if i > num_steps_burn_in:
                if i % FLAGS.print_steps == 0:
                    train_accuracy = accuracy.eval(feed_dict={
                        x: batch[0], y_: batch[1], keep_prob: 1.0})
                    print('%s: step %d, duration = %.3f, training accuracy %g' % (datetime.now(), i - num_steps_burn_in, duration, train_accuracy))
                total_duration += duration
                total_duration_squared += duration * duration

        mn = total_duration / FLAGS.num_batches
        vr = total_duration_squared / FLAGS.num_batches - mn * mn
        sd = math.sqrt(vr)
        print('%s: %s across %d steps, %.3f +/- %.3f sec / batch' % (datetime.now(), "Forward_backword", FLAGS.num_batches, mn, sd))
        print('Simple CNN test accuracy %g' % accuracy.eval(feed_dict={
            x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

def conv2d(x, W):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    return tf.get_variable(name='weight', shape=shape, initializer=tf.truncated_normal_initializer(stddev=0.1))

def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    return tf.get_variable(name='bias', shape=shape, initializer=tf.constant_initializer(0.1))

def get_names(graph=tf.get_default_graph()):
    return [t.name for op in graph.get_operations() for t in op.values()]

def get_tensors(graph=tf.get_default_graph()):
    return [t for op in graph.get_operations() for t in op.values()]

def dnn_model_fn(features, labels, mode):

    # Build the graph
    with tf.name_scope('fc'):
        W = tf.Variable(tf.zeros([784, 10]))
        b = tf.Variable(tf.zeros([10]))

        y = tf.matmul(features, W) + b

    predictions = {
        'classes': tf.argmax(input=y, axis=1, name='classes'),
        'probabilities': tf.nn.softmax(y, name='softmax_tensor')
    }

    if (mode == tf.estimator.ModeKeys.PREDICT):
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    with tf.name_scope('loss'):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=y))

    with tf.name_scope('accuracy'):
        accuracy = tf.metrics.accuracy(labels=tf.argmax(labels, 1), predictions=predictions['classes'], name='accuracy')
        batch_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(labels, 1), predictions['classes']), tf.float32))

    eval_metric_ops = {'accuracy': accuracy}

    tf.summary.scalar('accuracy', batch_accuracy)

    if (mode == tf.estimator.ModeKeys.EVAL):

        return tf.estimator.EstimatorSpec(mode=mode, loss=cross_entropy, eval_metric_ops=eval_metric_ops)

    if (mode == tf.estimator.ModeKeys.TRAIN):
        with tf.name_scope('GD_optimizer'):
            train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy, tf.train.get_global_step())
        
        return tf.estimator.EstimatorSpec(mode=mode, loss=cross_entropy, train_op=train_step, eval_metric_ops=eval_metric_ops)

def cnn_model_fn(features, labels, mode):

    labels = tf.to_int64(labels)

    # Reshape to use within a convolutional neural net.
    # Last dimension is for "features" - there is only one here, since images are
    # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
    with tf.name_scope('reshape'):
        x_image = tf.reshape(features, [-1, 28, 28, 1])

    # First convolutional layer - maps one grayscale image to 32 feature maps.
    with tf.variable_scope('conv1'):
        W_conv1 = weight_variable([5, 5, 1, 32])
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

    # Pooling layer - downsamples by 2X.
    with tf.name_scope('pool1'):
        h_pool1 = max_pool_2x2(h_conv1)

    # Second convolutional layer -- maps 32 feature maps to 64.
    with tf.variable_scope('conv2'):
        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    # Second pooling layer.
    with tf.name_scope('pool2'):
        h_pool2 = max_pool_2x2(h_conv2)

    # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
    # is down to 7x7x64 feature maps -- maps this to 1024 features.
    with tf.variable_scope('fc1'):
        W_fc1 = weight_variable([7 * 7 * 64, 1024])
        b_fc1 = bias_variable([1024])

        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Dropout - controls the complexity of the model, prevents co-adaptation of features.
    with tf.name_scope('dropout'):
        keep_prob = 0.5
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Map the 1024 features to 10 classes, one for each digit
    with tf.variable_scope('fc2'):
        W_fc2 = weight_variable([1024, 10])
        b_fc2 = bias_variable([10])
    
        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    
    predictions = {
        'classes': tf.argmax(input=y_conv, axis=1, name='classes'),
        'probabilities': tf.nn.softmax(y_conv, name='softmax_tensor')
    }

    if (mode == tf.estimator.ModeKeys.PREDICT):
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    with tf.name_scope('loss'):
        cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(
            labels=labels, logits=y_conv))

    with tf.name_scope('accuracy'):
        # correct_prediction = tf.equal(tf.argmax(y_conv, 1), labels)
        # correct_prediction = tf.cast(correct_prediction, tf.float32)
        # accuracy = tf.reduce_mean(correct_prediction)
        #accuracy = correct_prediction
        accuracy = tf.metrics.accuracy(labels=labels, predictions=predictions['classes'], name='accuracy')
        batch_accuracy = tf.reduce_mean(tf.cast(tf.equal(labels, predictions['classes']), tf.float32))
    
    eval_metric_ops = {
        'accuracy': accuracy
    }
    
    tf.summary.scalar('batch_accuracy', batch_accuracy)

    if (mode == tf.estimator.ModeKeys.EVAL):
        return tf.estimator.EstimatorSpec(mode=mode, loss=cross_entropy, eval_metric_ops=eval_metric_ops)
    
    if (mode == tf.estimator.ModeKeys.TRAIN):
        with tf.name_scope('adam_optimizer'):
            train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy, tf.train.get_global_step())

        return tf.estimator.EstimatorSpec(mode=mode, loss=cross_entropy, train_op=train_step, eval_metric_ops=eval_metric_ops)

def mnist_input_fn(batch_size, one_hot=False):
    mnist = input_data.read_data_sets('MNIST_data', one_hot=one_hot)

    while (True):
        batch = mnist.train.next_batch(batch_size)
        yield {'features':batch[0], 'labels':batch[1]}

# cnn False/ dnn True
def simple_estimator(model_fn, one_hot):
    work_estimator = tf.estimator.Estimator(model_fn=model_fn, model_dir="./est")
    work_estimator.train(lambda:mnist_input_fn(FLAGS.batch_size, one_hot), steps=1000)
    work_estimator.evaluate(lambda:mnist_input_fn(FLAGS.batch_size, one_hot), steps=10)

def main(_):

    program_start_time = time.time()
    # simple_dnn()
    # simple_cnn()

    simple_estimator(cnn_model_fn, False)

    program_end_time = time.time()
    print('Program finished, Total seconds: %s' % (program_end_time - program_start_time))

if __name__ == '__main__':
    tf.app.run()
