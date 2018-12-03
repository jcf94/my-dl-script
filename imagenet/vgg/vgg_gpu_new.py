import tensorflow.python.platform
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import data_flow_ops, state_ops, math_ops
from tensorflow.python.ops import variable_scope, control_flow_ops
from tensorflow.python.client import timeline
from datetime import datetime
import os
import math
import time
import six

# ----- CPU / GPU Set

#os.environ['CUDA_VISIBLE_DEVICES'] = '2'
CONFIG = tf.ConfigProto()
CONFIG.gpu_options.allow_growth=True
#CONFIG.log_device_placement=True

# -----

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch_size', 64,
                            """Batch size.""")
tf.app.flags.DEFINE_integer('num_batches', 100,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_string('data_format', 'NCHW',
                           """The data format for Convnet operations.
                           Can be either NHWC or NCHW.
                           """)
tf.app.flags.DEFINE_string('local_parameter_device', 'gpu',
                            """""")
tf.app.flags.DEFINE_string('job_name', None,
                            """""")
tf.app.flags.DEFINE_integer('task_index', 0,
                            """""")
tf.app.flags.DEFINE_integer('num_gpus', 1,
                            """""")
tf.app.flags.DEFINE_string('model', "vgg16",
                            """""")
tf.app.flags.DEFINE_boolean('staged_vars', False,
                            """""")
tf.app.flags.DEFINE_boolean('pipline_vars', False,
                            """""")
tf.app.flags.DEFINE_boolean('while_loop', False,
                            """ Use while_loop to run.""")
tf.app.flags.DEFINE_integer('loop_step', 10,
                            """""")
tf.app.flags.DEFINE_string('trace_file', None,
                            """""")
tf.app.flags.DEFINE_boolean('easgd', False,
                            """""")
tf.app.flags.DEFINE_integer('easgd_period', 10,
                            """""")

conv_counter = 0
pool_counter = 0
affine_counter = 0

class Vgg(object):
    def __init__(self, image_size):
        """Init"""
        self._conv_counter = 0
        self._pool_counter = 0
        self._affine_counter = 0
        self._image_size = image_size
        if FLAGS.data_format == 'NCHW':
            self._image_shape = [FLAGS.batch_size, 3, self._image_size, self._image_size]
        else:
            self._image_shape = [FLAGS.batch_size, self._image_size, self._image_size, 3]
        
        self.var_list = []
        self.var_counter = 0

    def _conv(self, var_list, inpOp, nIn, nOut, kH, kW, dH=1, dW=1, padType='SAME'):
        name = 'conv' + str(self._conv_counter)
        self._conv_counter += 1
        with tf.variable_scope(name):
            #kernel = tf.get_variable('weights', [kH, kW, nIn, nOut], tf.float32, tf.truncated_normal_initializer(stddev=1e-1), trainable=True)
            #biases = tf.get_variable('biases', [nOut], tf.float32, tf.zeros_initializer(), trainable=True)
            kernel = var_list[self.var_counter]
            self.var_counter = self.var_counter+1
            biases = var_list[self.var_counter]
            self.var_counter = self.var_counter+1

            if FLAGS.data_format == 'NCHW':
                strides = [1, 1, dH, dW]
            else:
                strides = [1, dH, dW, 1]
            conv = tf.nn.conv2d(inpOp, kernel, strides, padding=padType,
                                data_format=FLAGS.data_format)
            
            bias = tf.reshape(tf.nn.bias_add(conv, biases,
                                            data_format=FLAGS.data_format),
                            conv.get_shape())
            conv1 = tf.nn.relu(bias, name='relu')
            return conv1

    def _affine(self, var_list, inpOp, nIn, nOut, need_relu=True):
        name = 'affine' + str(self._affine_counter)
        self._affine_counter += 1
        with tf.variable_scope(name):
            #kernel = tf.get_variable('weights', [nIn, nOut], tf.float32, tf.truncated_normal_initializer(stddev=1e-1))
            #biases = tf.get_variable('biases', [nOut], tf.float32, tf.zeros_initializer())
            kernel = var_list[self.var_counter]
            self.var_counter = self.var_counter+1
            biases = var_list[self.var_counter]
            self.var_counter = self.var_counter+1

            xw_plus_b = tf.nn.xw_plus_b(inpOp, kernel, biases, name='xw_plus_b')
            if need_relu:
                affine1 = tf.nn.relu(xw_plus_b)
                return affine1
            else:
                return xw_plus_b

    def _mpool(self, inpOp, kH, kW, dH=2, dW=2):
        name = 'pool' + str(self._pool_counter)
        self._pool_counter += 1
        if FLAGS.data_format == 'NCHW':
            ksize = [1, 1, kH, kW]
            strides = [1, 1, dH, dW]
        else:
            ksize = [1, kH, kW, 1]
            strides = [1, dH, dW, 1]
        
        with tf.name_scope(name):
            pool = tf.nn.max_pool(inpOp,
                            ksize=ksize,
                            strides=strides,
                            padding='VALID',
                            data_format=FLAGS.data_format,
                            name='MaxPool')

        return pool

    def loss(self, logits, labels):
        with tf.name_scope('xentropy'):
            cross_entropy = tf.losses.sparse_softmax_cross_entropy(logits=logits,
                                                                labels=labels)
            loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
        return loss

    def _construct_vgg(self, images, num_conv_layers, var_list):
        """Build vgg architecture from blocks."""
        assert len(num_conv_layers) == 5
        self.var_counter = 0
        last = images
        last_out = 3
        for _ in range(num_conv_layers[0]):
            last = self._conv(var_list, last, last_out, 64, 3, 3)
            last_out = 64
        last = self._mpool(last, 2, 2)
        for _ in range(num_conv_layers[1]):
            last = self._conv(var_list, last, last_out, 128, 3, 3)
            last_out = 128
        last = self._mpool(last, 2, 2)
        for _ in range(num_conv_layers[2]):
            last = self._conv(var_list, last, last_out, 256, 3, 3)
            last_out = 256
        last = self._mpool(last, 2, 2)
        for _ in range(num_conv_layers[3]):
            last = self._conv(var_list, last, last_out, 512, 3, 3)
            last_out = 512
        last = self._mpool(last, 2, 2)
        for _ in range(num_conv_layers[4]):
            last = self._conv(var_list, last, last_out, 512, 3, 3)
            last_out = 512
        last = self._mpool(last, 2, 2)
        last = tf.reshape(last, [-1, 512 * 7 * 7])
        last = self._affine(var_list, last, 512 * 7 * 7, 4096)
        last = self._affine(var_list, last, 4096, 4096)
        last = self._affine(var_list, last, 4096, 1000, False)

        return last

    def inference(self, images, var_list):
        with tf.variable_scope('cg', reuse=tf.AUTO_REUSE):
            if FLAGS.model == 'vgg11':
                return self._construct_vgg(images, [1, 1, 2, 2, 2], var_list)

            if FLAGS.model == 'vgg16':
                return self._construct_vgg(images, [2, 2, 3, 3, 3], var_list)

            if FLAGS.model == 'vgg19':
                return self._construct_vgg(images, [2, 2, 4, 4, 4], var_list)

    def _conv_var(self, nIn, nOut, kH, kW, dH=1, dW=1, padType='SAME'):
        name = 'conv' + str(self._conv_counter)
        self._conv_counter += 1
        with tf.variable_scope(name):
            kernel = tf.get_variable('weights', [kH, kW, nIn, nOut], tf.float32, tf.truncated_normal_initializer(stddev=1e-1), trainable=True)
            biases = tf.get_variable('biases', [nOut], tf.float32, tf.zeros_initializer(), trainable=True)
        
        self.var_list.append(kernel)
        self.var_list.append(biases)

    def _affine_var(self, nIn, nOut, need_relu=True):
        name = 'affine' + str(self._affine_counter)
        self._affine_counter += 1
        with tf.variable_scope(name):
            kernel = tf.get_variable('weights', [nIn, nOut], tf.float32, tf.truncated_normal_initializer(stddev=1e-1))
            biases = tf.get_variable('biases', [nOut], tf.float32, tf.zeros_initializer())
        
        self.var_list.append(kernel)
        self.var_list.append(biases)

    def _construct_vgg_var(self, num_conv_layers):
        """Build vgg architecture from blocks."""
        assert len(num_conv_layers) == 5
        last_out = 3
        for _ in range(num_conv_layers[0]):
            self._conv_var(last_out, 64, 3, 3)
            last_out = 64
        for _ in range(num_conv_layers[1]):
            self._conv_var(last_out, 128, 3, 3)
            last_out = 128
        for _ in range(num_conv_layers[2]):
            self._conv_var(last_out, 256, 3, 3)
            last_out = 256
        for _ in range(num_conv_layers[3]):
            self._conv_var(last_out, 512, 3, 3)
            last_out = 512
        for _ in range(num_conv_layers[4]):
            self._conv_var(last_out, 512, 3, 3)
            last_out = 512
        self._affine_var(512 * 7 * 7, 4096)
        self._affine_var(4096, 4096)
        self._affine_var(4096, 1000, False)

    def prepare_var(self):
        with tf.variable_scope('cg', reuse=tf.AUTO_REUSE):
            if FLAGS.model == 'vgg11':
                return self._construct_vgg_var([1, 1, 2, 2, 2])

            if FLAGS.model == 'vgg16':
                return self._construct_vgg_var([2, 2, 3, 3, 3])

            if FLAGS.model == 'vgg19':
                return self._construct_vgg_var([2, 2, 4, 4, 4])

class BenchMark(object):
    def __init__(self):
        """Init"""
        if FLAGS.job_name:
            self.worker_prefix = '/job:worker/task:%s' % FLAGS.task_index
        else:
            self.worker_prefix = ''
        
        self.cpu_device = '%s/cpu:0' % self.worker_prefix
        self.gpu_devices = [
            '%s/%s:%i' % (self.worker_prefix, 'gpu', i)
            for i in range(FLAGS.num_gpus)
        ]
        if FLAGS.local_parameter_device == 'gpu':
            self.param_server_device = self.gpu_devices[0]
        else:
            self.param_server_device = self.cpu_device

        self.replica_devices = [
            tf.train.replica_device_setter(
                worker_device=d,
                ps_device=self.param_server_device,
                ps_tasks=1) for d in self.gpu_devices
        ]

        self.global_step_device = self.param_server_device
        self.v_mgr = None

    def build_network(self, image_size):
        grads_list = []

        for device_index, device_name in enumerate(self.replica_devices):
            network = Vgg(image_size)
            with tf.name_scope('tower_%i' % device_index):
                # -------------------------------------------------------------------
                with tf.device(self.gpu_devices[device_index]), tf.variable_scope('Gpu_%i_Own' % device_index, reuse=tf.AUTO_REUSE):
                    images = tf.get_variable('gpu_cache_images', network._image_shape, tf.float32,
                                            tf.truncated_normal_initializer(1e-1), trainable=False)
                    labels = tf.random_uniform(
                        [FLAGS.batch_size],
                        minval=0,
                        maxval=1000,
                        dtype=tf.int32,
                        name='synthetic_labels')
                # -------------------------------------------------------------------
                with tf.device(device_name):
                    network.prepare_var()
                    varis = network.var_list
                # -------------------------------------------------------------------
                with tf.device(self.gpu_devices[device_index]):
                    last_layer = network.inference(images, varis)
                    loss = network.loss(last_layer, labels)

                    grads = tf.gradients(loss, varis, aggregation_method=tf.AggregationMethod.DEFAULT)

                    grads_list.append(grads)
                # -------------------------------------------------------------------

        # -------------------------------------------------------------------

        with tf.device(self.param_server_device):
            if FLAGS.num_gpus > 1:
                average_grads = []
                for grads in zip(*grads_list):
                    average_grads.append(tf.multiply(tf.add_n(grads), 1.0 / FLAGS.num_gpus))
                grads_and_varis = list(zip(average_grads, varis))
            else:
                grads_and_varis = list(zip(grads_list[0], varis))
            optimizer = tf.train.GradientDescentOptimizer(0.01).apply_gradients(grads_and_varis, tf.train.get_or_create_global_step())

        return optimizer

    def build_loop_network(self, image_size, loop_step):

        print('|------ Build While_loop Network')

        for device_index, device_name in enumerate(self.replica_devices):
            network = Vgg(image_size)
            with tf.name_scope('tower_%i' % device_index):
                # -------------------------------------------------------------------
                with tf.device(self.gpu_devices[device_index]), tf.variable_scope('Gpu_%i_Own' % device_index, reuse=tf.AUTO_REUSE):
                    images = tf.get_variable('gpu_cache_images', network._image_shape, tf.float32,
                                            tf.truncated_normal_initializer(1e-1), trainable=False)
                    labels = tf.random_uniform(
                        [FLAGS.batch_size],
                        minval=0,
                        maxval=1000,
                        dtype=tf.int32,
                        name='synthetic_labels')
                # -------------------------------------------------------------------
                with tf.device(device_name):
                    network.prepare_var()
                    varis = network.var_list
                # -------------------------------------------------------------------

                def cond(iter_step, *var_list):
                    return iter_step < loop_step
                
                def body(iter_step, *var_list):
                    var_list = list(var_list)
                    out_list = []

                    last_layer = network.inference(images, var_list)
                    loss = network.loss(last_layer, labels)
                    grads = tf.gradients(loss, var_list, aggregation_method=tf.AggregationMethod.DEFAULT)

                    grads_and_varis = list(zip(grads, network.var_list))
                    optimizer = tf.train.GradientDescentOptimizer(0.01)
                    for grad_var in grads_and_varis:
                        train_op = optimizer.apply_gradients([grad_var])
                        with tf.control_dependencies([train_op]):
                            ide = tf.identity(grad_var[1])
                            out_list.append(ide)
                    
                    return [iter_step+1, *out_list]

                with tf.device(self.gpu_devices[device_index]):
                    loop_op = tf.while_loop(cond, body, [tf.constant(0), *varis], parallel_iterations=1)
                # -------------------------------------------------------------------

        return loop_op

    def do_step_run(self, image_size):
        print('|------ Start Per_step Run')

        optimizer = self.build_network(image_size)

        init_op = [tf.global_variables_initializer(), tf.local_variables_initializer()]

        with tf.Session(config=CONFIG) as sess:

            tf.summary.FileWriter("./", sess.graph)

            num_steps_burn_in = 10
            total_duration = 0.0
            total_duration_squared = 0.0
            sess.run(init_op)

            for i in range(FLAGS.num_batches + num_steps_burn_in):
                start_time = time.time()
                _ = sess.run(optimizer)
                duration = time.time() - start_time
                if i > num_steps_burn_in:
                    if not i % 10:
                        picps = FLAGS.num_gpus * FLAGS.batch_size / duration
                        print('%s: step %d, duration = %.3f, speed = %.3f pics/s' %
                            (datetime.now(), i - num_steps_burn_in, duration, picps))
                    total_duration += duration
                    total_duration_squared += duration * duration

            mn = total_duration / FLAGS.num_batches
            vr = total_duration_squared / FLAGS.num_batches - mn * mn
            sd = math.sqrt(vr)
            print('%s: %s across %d steps, %.3f +/- %.3f sec / batch' %
                (datetime.now(), "Forward_backword", FLAGS.num_batches, mn, sd))
            picps = (FLAGS.num_gpus * FLAGS.num_batches * FLAGS.batch_size) / total_duration
            print('%.3f pics/s' % picps)

            if FLAGS.trace_file:
                options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                _ = sess.run(optimizer, options=options, run_metadata=run_metadata)
                fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                chrome_trace = fetched_timeline.generate_chrome_trace_format()
                with open(FLAGS.trace_file, 'w') as f:
                    f.write(chrome_trace)
                print('Chrome Trace File write in %s' % FLAGS.trace_file)

    def do_while_run(self, image_size):
        print('|------ Start While_loop Run')

        optimizer = self.build_loop_network(image_size, FLAGS.loop_step)

        init_op = [tf.global_variables_initializer(), tf.local_variables_initializer()]

        with tf.Session(config=CONFIG) as sess:

            tf.summary.FileWriter("./", sess.graph)

            num_steps_burn_in = 20
            total_duration = 0.0
            total_duration_squared = 0.0
            sess.run(init_op)

            for i in range(int((FLAGS.num_batches + num_steps_burn_in) / 10)):
                start_time = time.time()
                _ = sess.run(optimizer)
                duration = time.time() - start_time
                if i >= num_steps_burn_in/10:
                    picps = FLAGS.num_gpus * FLAGS.batch_size * FLAGS.loop_step / duration
                    print('%s: step %d, duration = %.3f, speed = %.3f pics/s' %
                        (datetime.now(), i*10 - num_steps_burn_in, duration, picps))
                    total_duration += duration
                    total_duration_squared += duration * duration

            mn = total_duration / FLAGS.num_batches
            vr = total_duration_squared / FLAGS.num_batches - mn * mn
            sd = math.sqrt(vr)
            print('%s: %s across %d steps, %.3f +/- %.3f sec / batch' %
                (datetime.now(), "Forward_backword", FLAGS.num_batches, mn, sd))
            picps = (FLAGS.num_gpus * FLAGS.num_batches * FLAGS.batch_size) / total_duration
            print('%.3f pics/s' % picps)

            if FLAGS.trace_file:
                options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                _ = sess.run(optimizer, options=options, run_metadata=run_metadata)
                fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                chrome_trace = fetched_timeline.generate_chrome_trace_format()
                with open(FLAGS.trace_file, 'w') as f:
                    f.write(chrome_trace)
                print('Chrome Trace File write in %s' % FLAGS.trace_file)

def run_benchmark():

    image_size = 224

    bench = BenchMark()

    # ----------------------- Fake Input Images -----------------------

    with tf.device(bench.cpu_device), tf.name_scope('Fake_Input_Images'):
        if FLAGS.data_format == 'NCHW':
            image_shape = [FLAGS.batch_size, 3, image_size, image_size]
        else:
            image_shape = [FLAGS.batch_size, image_size, image_size, 3]
        ori_images = tf.Variable(tf.random_normal(image_shape,
                                                dtype=tf.float32,
                                                stddev=1e-1), trainable=False)

        ori_labels = tf.Variable(tf.ones([FLAGS.batch_size], dtype=tf.int64), trainable=False)

    # -----------------------------------------------------------------

    if FLAGS.while_loop:
        return bench.do_while_run(image_size)
    else:
        return bench.do_step_run(image_size)

def main(_):
    program_start_time = time.time()
    run_benchmark()
    program_end_time = time.time()
    print('Program finished, Total seconds: %s' % (program_end_time - program_start_time))

if __name__ == '__main__':
    tf.app.run()
