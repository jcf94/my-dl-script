import tensorflow.python.platform
import tensorflow as tf
from datetime import datetime
import os
import math
import time

# ----- CPU / GPU Set

os.environ['CUDA_VISIBLE_DEVICES'] = '7'
CONFIG = tf.ConfigProto()
CONFIG.gpu_options.allow_growth=True
#CONFIG.log_device_placement=True

# -----

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch_size', 24,
                            """Batch size.""")
tf.app.flags.DEFINE_integer('num_batches', 300,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_string('data_format', 'NHWC',
                           """The data format for Convnet operations.
                           Can be either NHWC or NCHW.
                           """)

conv_counter = 1
pool_counter = 1
affine_counter = 1

def _conv(inpOp, nIn, nOut, kH, kW, dH, dW, padType):
    global conv_counter
    name = 'conv' + str(conv_counter)
    conv_counter += 1
    with tf.name_scope(name) as scope:
        kernel = tf.Variable(tf.truncated_normal([kH, kW, nIn, nOut],
                                                 dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        if FLAGS.data_format == 'NCHW':
            strides = [1, 1, dH, dW]
        else:
            strides = [1, dH, dW, 1]
        conv = tf.nn.conv2d(inpOp, kernel, strides, padding=padType,
                            data_format=FLAGS.data_format)
        biases = tf.Variable(tf.constant(0.0, shape=[nOut], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.reshape(tf.nn.bias_add(conv, biases,
                                         data_format=FLAGS.data_format),
                          conv.get_shape())
        conv1 = tf.nn.relu(bias, name=scope)
        return conv1

def _affine(inpOp, nIn, nOut):
    global affine_counter
    name = 'affine' + str(affine_counter)
    affine_counter += 1
    with tf.name_scope(name) as scope:
        kernel = tf.Variable(tf.truncated_normal([nIn, nOut],
                                                 dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        biases = tf.Variable(tf.constant(0.0, shape=[nOut], dtype=tf.float32),
                             trainable=True, name='biases')
        affine1 = tf.nn.relu_layer(inpOp, kernel, biases, name=name)
        return affine1

def _mpool(inpOp, kH, kW, dH, dW):
    global pool_counter
    name = 'pool' + str(pool_counter)
    pool_counter += 1
    if FLAGS.data_format == 'NCHW':
        ksize = [1, 1, kH, kW]
        strides = [1, 1, dH, dW]
    else:
        ksize = [1, kH, kW, 1]
        strides = [1, dH, dW, 1]
    return tf.nn.max_pool(inpOp,
                          ksize=ksize,
                          strides=strides,
                          padding='VALID',
                          data_format=FLAGS.data_format,
                          name=name)

def loss(logits, labels):
    with tf.name_scope("loss") as scope:
        batch_size = tf.size(labels)

        labels = tf.expand_dims(labels, 1)
        # labels: [24, 1]

        indices = tf.expand_dims(tf.range(0, batch_size, 1), 1)
        # indices: [24, 1]

        concated = tf.concat([indices, labels], 1)
        # concated: [24, 2]

        onehot_labels = tf.sparse_to_dense(
            concated, tf.stack([batch_size, 1000]), 1.0, 0.0)
        # onehot_labels: [24, 1000]

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                                labels=onehot_labels,
                                                                name='xentropy')
        loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
        return loss

def inference(images):
    conv1 = _conv (images, 3, 96, 11, 11, 4, 4, 'VALID')
    pool1 = _mpool(conv1,  3, 3, 2, 2)
    conv2 = _conv (pool1,  96, 256, 5, 5, 1, 1, 'SAME')
    pool2 = _mpool(conv2,  3, 3, 2, 2)
    conv3 = _conv (pool2,  256, 384, 3, 3, 1, 1, 'SAME')
    conv4 = _conv (conv3,  384, 256, 3, 3, 1, 1, 'SAME')
    conv5 = _conv (conv4,  256, 256, 3, 3, 1, 1, 'SAME')
    pool5 = _mpool(conv5,  3, 3, 2, 2)
    resh1 = tf.reshape(pool5, [-1, 256 * 6 * 6])
    affn1 = _affine(resh1, 256 * 6 * 6, 4096)
    affn2 = _affine(affn1, 4096, 4096)
    affn3 = _affine(affn2, 4096, 1000)
    return affn3

def run_benchmark():
    image_size = 224
    if FLAGS.data_format == 'NCHW':
        image_shape = [FLAGS.batch_size, 3, image_size + 3, image_size + 3]
    else:
        image_shape = [FLAGS.batch_size, image_size + 3, image_size + 3, 3]
    images = tf.Variable(tf.random_normal(image_shape,
                                              dtype=tf.float32,
                                              stddev=1e-1))

    labels = tf.Variable(tf.ones([FLAGS.batch_size], dtype=tf.int32))

    last_layer = inference(images)

    objective = loss(last_layer, labels)

    optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(objective)

    init = tf.global_variables_initializer()

    with tf.Session(config=CONFIG) as sess:
        num_steps_burn_in = 10
        total_duration = 0.0
        total_duration_squared = 0.0
        sess.run(init)
        for i in range(FLAGS.num_batches + num_steps_burn_in):
            start_time = time.time()
            _ = sess.run(optimizer)
            duration = time.time() - start_time
            if i > num_steps_burn_in:
                if not i % 10:
                    print('%s: step %d, duration = %.3f' %
                        (datetime.now(), i - num_steps_burn_in, duration))
                total_duration += duration
                total_duration_squared += duration * duration

        mn = total_duration / FLAGS.num_batches
        vr = total_duration_squared / FLAGS.num_batches - mn * mn
        sd = math.sqrt(vr)
        print('%s: %s across %d steps, %.3f +/- %.3f sec / batch' %
            (datetime.now(), "Forward_backword", FLAGS.num_batches, mn, sd))

def main(_):
    program_start_time = time.time()
    run_benchmark()
    program_end_time = time.time()
    print('Program finished, Total seconds: %s' % (program_end_time - program_start_time))

if __name__ == '__main__':
    tf.app.run()
