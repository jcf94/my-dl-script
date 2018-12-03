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

from vgg.model import Vgg

# ----- CPU / GPU Set

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
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
                            """ Use while_loop to run.""")
tf.app.flags.DEFINE_string('trace_file', None,
                            """""")

class VariableMgrLocalFetchFromPS(object):
    def __init__(self, bench):
        self.bench = bench
        self.gpu_devices = bench.gpu_devices
        self.vars_on_devices = [
            dict() for _ in self.gpu_devices
        ]

    def trainable_variables_on_device(self,
                                    rel_device_num,
                                    writable=False):
        return self.custom_getter.trainable_variables_on_device(
            rel_device_num, writable=writable)

    def create_outer_variable_scope(self, device_num):
        self.custom_getter = VariableGetter(self, device_num)
        return tf.variable_scope(
            'v', reuse=bool(device_num), custom_getter=self.custom_getter)

class VariableGetter(object):
    def __init__(self, variable_mgr, device_num):
        self.variable_mgr = variable_mgr
        self.device_num = device_num
    
    def __call__(self, getter, name, *args, **kwargs):
        vars_on_devices = self.variable_mgr.vars_on_devices[self.device_num]

        if name in vars_on_devices:
            real_var = vars_on_devices[name]
            get_op = tf.identity(real_var)
            return get_op

        with tf.device(self.variable_mgr.bench.param_server_device):
            real_var = getter(name, *args, **kwargs)

        trainable = kwargs['trainable']

        if trainable:
            vars_on_devices[name] = real_var
            return tf.identity(real_var)
        else:
            return real_var

    def trainable_variables_on_device(self, rel_device_num,
                                        writable):
        params_refs = tf.trainable_variables()
        if writable:
            return params_refs
        params = []
        for param in params_refs:
            var_name = param.name.split(':')[0]
            _, var_get_op = self.variable_mgr.staging_vars_on_devices[rel_device_num][
                var_name]
            params.append(var_get_op)
        return params

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
        if not self.v_mgr:
            self.v_mgr = VariableMgrLocalFetchFromPS(self)
        v_mgr = self.v_mgr

        for device_index, device_name in enumerate(self.replica_devices):
            network = Vgg(image_size, FLAGS.data_format, FLAGS.batch_size, FLAGS.model)
            with tf.name_scope('tower_%i' % device_index):
                # -------------------------------------------------------------------

                with tf.device(self.gpu_devices[device_index]), tf.variable_scope('Gpu_%i_Own' % device_index, reuse=tf.AUTO_REUSE):
                    images = tf.get_variable('gpu_cache_images', network._image_shape, tf.float32,
                                            tf.truncated_normal_initializer(1e-1), trainable=False)

                    #labels = tf.Variable(tf.ones([FLAGS.batch_size], dtype=tf.int64), trainable=False, name='gpu_cache_labels')
                    labels = tf.random_uniform(
                        [FLAGS.batch_size],
                        minval=0,
                        maxval=1000,
                        dtype=tf.int32,
                        name='synthetic_labels')

                # -------------------------------------------------------------------
                with tf.device(device_name), v_mgr.create_outer_variable_scope(device_index):
                    last_layer = network.inference(images)
                    loss = network.loss(last_layer, labels)
                    varis = tf.trainable_variables()
                    grads = tf.gradients(loss, varis, aggregation_method=tf.AggregationMethod.DEFAULT)

                    grads_list.append(grads)

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

    def do_step_run(self, image_size):
        print('|------ Start Per_step Run')

        optimizer = self.build_network(image_size)
        enqueue_ops = None

        init_op = [tf.global_variables_initializer(), tf.local_variables_initializer()]

        with tf.Session(config=CONFIG) as sess:

            tf.summary.FileWriter("./", sess.graph)

            num_steps_burn_in = 10
            total_duration = 0.0
            total_duration_squared = 0.0
            sess.run(init_op)

            if FLAGS.easgd:
                sess.run(easgd_prepare)

            if FLAGS.staged_vars:
                for i in range(len(enqueue_ops)):
                    sess.run(enqueue_ops[:(i + 1)])

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

    return bench.do_step_run(image_size)

def main(_):
    program_start_time = time.time()
    run_benchmark()
    program_end_time = time.time()
    print('Program finished, Total seconds: %s' % (program_end_time - program_start_time))

if __name__ == '__main__':
    tf.app.run()
